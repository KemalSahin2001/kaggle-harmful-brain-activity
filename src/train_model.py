import numpy as np
from sklearn.model_selection import GroupKFold


def train_model(
    train,
    DataGenerator,
    strategy,
    build_model_fn,
    EPOCHS,
    LR_scheduler,
    TARGETS,
    USE_KAGGLE_SPECTROGRAMS,
    USE_EEG_SPECTROGRAMS,
    specs,
    eeg_specs,
    version: str = "b0",
    LOAD_MODELS_FROM=None,
    K_FOLDS=5,  # New parameter for number of folds
    SAMPLE_PERCENT=100,  # New parameter for dataset sampling percentage
):
    # Sample the dataset
    sampled_train = train.sample(frac=SAMPLE_PERCENT / 100.0)

    all_oof = []
    all_true = []

    gkf = GroupKFold(n_splits=K_FOLDS)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(sampled_train, sampled_train.target, sampled_train.patient_id)
    ):
        print(f"### Fold {i+1}/{K_FOLDS} with {SAMPLE_PERCENT}% of data")

        train_gen = DataGenerator(
            sampled_train.iloc[train_index],
            shuffle=True,
            batch_size=32,
            augment=False,
            TARGETS=TARGETS,
            specs=specs,
            eeg_specs=eeg_specs,
        )
        valid_gen = DataGenerator(
            sampled_train.iloc[valid_index],
            shuffle=False,
            batch_size=64,
            mode="valid",
            TARGETS=TARGETS,
            specs=specs,
            eeg_specs=eeg_specs,
        )

        with strategy.scope():
            model = build_model_fn(
                USE_KAGGLE_SPECTROGRAMS, USE_EEG_SPECTROGRAMS, version=version
            )

        if LOAD_MODELS_FROM is None:
            model.fit(
                train_gen,
                verbose=1,
                validation_data=valid_gen,
                epochs=EPOCHS,
                callbacks=[LR_scheduler],
            )
            model.save_weights(
                f"EffNet_v{version}_f{i}_p{SAMPLE_PERCENT}_k{K_FOLDS}.weights.h5"
            )
        else:
            model.load_weights(
                f"{LOAD_MODELS_FROM}EffNet_v{version}_f{i}_p{SAMPLE_PERCENT}_k{K_FOLDS}.weights.h5"
            )

        oof = model.predict(valid_gen, verbose=1)
        all_oof.append(oof)
        all_true.append(sampled_train.iloc[valid_index][TARGETS].values)

        del model, oof

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    return all_oof, all_true
