import gc

import numpy as np
from sklearn.model_selection import GroupKFold


def train_model(
    train,
    DataGenerator,
    strategy,
    build_model_fn,
    VER,
    EPOCHS,
    LR,
    TARGETS,
    USE_KAGGLE_SPECTROGRAMS,
    USE_EEG_SPECTROGRAMS,
    specs,
    eeg_specs,
    LOAD_MODELS_FROM=None,
):
    all_oof = []
    all_true = []

    gkf = GroupKFold(n_splits=5)
    for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        print("#" * 25)
        print(f"### Fold {i+1}")

        # Ensure TARGETS is correctly passed here
        train_gen = DataGenerator(
            train.iloc[train_index],
            shuffle=True,
            batch_size=32,
            augment=False,
            TARGETS=TARGETS,
            specs=specs,
            eeg_specs=eeg_specs,
        )
        valid_gen = DataGenerator(
            train.iloc[valid_index],
            shuffle=False,
            batch_size=64,
            mode="valid",
            TARGETS=TARGETS,
            specs=specs,
            eeg_specs=eeg_specs,
        )

        print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
        print("#" * 25)

        with strategy.scope():
            # Now calling build_model with its parameters inside train_model
            model = build_model_fn(USE_KAGGLE_SPECTROGRAMS, USE_EEG_SPECTROGRAMS)
        if LOAD_MODELS_FROM is None:
            model.fit(
                train_gen,
                verbose=1,
                validation_data=valid_gen,
                epochs=EPOCHS,
                callbacks=[LR],
            )
            model.save_weights(f"EffNet_v{VER}_f{i}.h5")
        else:
            model.load_weights(f"{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5")

        oof = model.predict(valid_gen, verbose=1)
        all_oof.append(oof)
        all_true.append(train.iloc[valid_index][TARGETS].values)

        del model, oof
        gc.collect()

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)

    return all_oof, all_true
