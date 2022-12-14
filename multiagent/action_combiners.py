import numpy as np

# Здесь представлены функции для объединения действий от разных агентов в общее
# Самый простой - для случая, когда не используются векторные среды. В таком случае для всех видов пространств
# действий можем просто соединить действия
NON_VEC_COMBINER = lambda acts: np.concatenate(acts)
# Обработка случая, когда модели используют дискретное пространство действий. В таком случае actions - уже
# массив чисел, который нужно вернуть. Concatenate для него вызовет ошибку.
NON_VEC_DISCRETE_COMBINER = lambda acts: acts
# Для случая, когда используется непрерывное пространство. Необходимо перевести (Число агентов, Число сред, 1, y)
# в (Число сред, Число Агентов, y)
BOX_COMBINER = lambda acts: np.concatenate(acts, axis=1)
# Для случая дискретных пространств. Переводит (Число агентов, Число сред, 1) в (Число сред, Число агентов, 1)
DISCRETE_COMBINER = lambda acts: np.array(acts).transpose()
#TODO: Понять как работает мульти-дискретный случай
MULTI_DISCRETE_COMBINER = lambda acts: np.concatenate(acts, axis=1)