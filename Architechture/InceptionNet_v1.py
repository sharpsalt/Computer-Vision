import tensorflow as tf
from tensorflow.keras import layers, Model


class Inception(tf.keras.layers.Layer):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # Branch 1: 1x1 conv
        self.branch1 = tf.keras.Sequential([
            layers.Conv2D(ch1x1, kernel_size=1, strides=1, padding='valid', activation='relu')
        ])

        # Branch 2: 1x1 reduction + 3x3
        self.branch2 = tf.keras.Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, strides=1, padding='valid', activation='relu'),
            layers.Conv2D(ch3x3, kernel_size=3, strides=1, padding='same', activation='relu')
        ])

        # Branch 3: 1x1 reduction + 5x5
        self.branch3 = tf.keras.Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, strides=1, padding='valid', activation='relu'),
            layers.Conv2D(ch5x5, kernel_size=5, strides=1, padding='same', activation='relu')
        ])

        # Branch 4: 3x3 max pool + 1x1 projection
        self.branch4 = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=3, strides=1, padding='same'),
            layers.Conv2D(pool_proj, kernel_size=1, strides=1, padding='valid', activation='relu')
        ])

    def call(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return tf.concat([b1, b2, b3, b4], axis=-1)


class InceptionAux(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_classes=1000):
        super(InceptionAux, self).__init__()
        self.avgpool = layers.AveragePooling2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu')
        self.dropout = layers.Dropout(0.7)
        self.fc2 = layers.Dense(num_classes)

    def call(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(Model):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        # Stem (Section 5 of the paper)
        self.conv1 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu'),  # 224→112
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')                     # 112→56
        ])

        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=1, strides=1, padding='valid', activation='relu'),  # 1×1 reduction
            layers.Conv2D(192, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')                       # 56→28
        ])

        # Inception blocks (exact channel counts from Table 1 of the paper)
        self.inception3a = Inception(192,  64,  96, 128, 16,  32,  32)
        self.inception3b = Inception(256, 128, 128, 192, 32,  96,  64)

        self.maxpool3 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')  # 28→14

        self.inception4a = Inception(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = Inception(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = Inception(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = Inception(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')  # 14→7

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # Final layers
        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.4)
        self.fc = layers.Dense(num_classes)

        # Auxiliary classifiers (attached to 4a and 4d)
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)   # after 4a
            self.aux2 = InceptionAux(528, num_classes)   # after 4d

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if training and self.aux_logits else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if training and self.aux_logits else None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)

        if training and self.aux_logits:
            return x, aux1, aux2
        return x

if __name__ == "__main__":
    model = GoogLeNet(num_classes=1000, aux_logits=True)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()  # Print model summary to verify
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = model(dummy_input, training=False)
    print("Main output shape :", output[0].shape if isinstance(output, tuple) else output.shape)
    print("   - 22 layers deep (27 with pools)")
    print("   - Auxiliary classifiers with weight 0.3 during training (implement in your loss function)") #Like it depends on ourself but it is necessary to give less weightage to auxilary and more weight to main architecture
    print("   - ~6.8M parameters")
