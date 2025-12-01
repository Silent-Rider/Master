import datetime
import os

from keras.src.callbacks import Callback


class BestModelsCallback(Callback):

    FOLDER = 'models/'

    def __init__(self, filepath_prefix, monitor='val_loss', mode='auto', max_models=3):
        super().__init__()
        self.filepath_prefix = filepath_prefix
        self.monitor = monitor

        if mode == 'auto':
            mode = 'min' if 'loss' in monitor else 'max'
        self.mode = mode
        self.best_scores = []
        self.max_models = max_models


    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        filepath = f"{BestModelsCallback.FOLDER}{date_str}_{self.filepath_prefix}_epoch_{epoch}_№{{}}.h5"

        # Определяем, сохранять ли текущую модель (есть ли место)
        if len(self.best_scores) < self.max_models:
            index = len(self.best_scores) + 1
            model_path = filepath.format(index)
            self.model.save(model_path)
            self.best_scores.append((current_score, model_path, index))
        else:
            # Уже собрано 3 модели — проверим, лучше ли текущая худшей из ТОП-3
            self.best_scores.sort(key=lambda x: x[0], reverse=(self.mode == 'min'))
            worst_score, worst_path, worst_index = self.best_scores[0]  # худшая в ТОП-3

            is_better = (
                    (self.mode == 'min' and current_score < worst_score) or
                    (self.mode == 'max' and current_score > worst_score)
            )

            if is_better:
                print()
                print(f'Found better model, {self.monitor} {current_score} at epoch {epoch}')
                # Удаляем худшую модель
                if os.path.exists(worst_path):
                    os.remove(worst_path)

                # Сохраняем новую на её место (меняем индекс на индекс удаленной модели)
                new_path = filepath.format(worst_index)
                self.model.save(filepath=new_path)
                self.best_scores[0] = (current_score, new_path, worst_index)