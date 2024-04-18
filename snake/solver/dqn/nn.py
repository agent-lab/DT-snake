import tensorflow as tf
class MyModel(tf.keras.Model):
    def __init__(self, num_visual_features, shape_visual_state, num_important_features, num_actions, use_visual_only, use_dueling, w_init, b_init):
        super(MyModel, self).__init__()
        self.num_visual_features = num_visual_features
        self.shape_visual_state = shape_visual_state
        self.num_important_features = num_important_features
        self.num_actions = num_actions
        self.use_visual_only = use_visual_only
        self.use_dueling = use_dueling
        self.w_init = w_init
        self.b_init = b_init

        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='valid', activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=2, strides=1, padding='valid', activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)

        # Dense layers
        self.fc1 = tf.keras.layers.Dense(units=1024, activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.fc2_v = tf.keras.layers.Dense(units=512, activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.fc2_a = tf.keras.layers.Dense(units=512, activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.v = tf.keras.layers.Dense(units=1, activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)
        self.a = tf.keras.layers.Dense(units=num_actions, activation=self.leaky_relu, kernel_initializer=w_init, bias_initializer=b_init)

    def call(self, inputs):
        # Slice and reshape operations
        visual_state = tf.slice(inputs, begin=[0, 0], size=[-1, self.num_visual_features])
        visual_state_2d = tf.reshape(visual_state, shape=[-1, *self.shape_visual_state])

        # Convolutional operations
        x = self.conv1(visual_state_2d)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = tf.reshape(x, shape=[-1, 2 * 2 * 256])  # Reshape after last conv layer

        if not self.use_visual_only:
            important_state = tf.slice(inputs, begin=[0, self.num_visual_features], size=[-1, self.num_important_features])
            x = tf.concat([x, important_state], axis=1)

        x = self.fc1(x)

        if self.use_dueling:
            v = self.v(self.fc2_v(x))
            a = self.a(self.fc2_a(x))
            a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
            q_all = v + (a - a_mean)
        else:
            x = self.fc1(x)
            q_all = self.fc2(x)
            q_all = self.a(q_all)

        return q_all

    def leaky_relu(self, x):
        return tf.nn.leaky_relu(x, alpha=0.01)