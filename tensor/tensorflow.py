import tensorflow as tf


class Tensor:
    def __int__(self):
        pass

    def ten(self):
        pass


if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)
