# model_hidden_128_bert_256_lang_32_cast_32_crew_32_batch_32: [1.7955729961395264, 1.783509373664856, 1.727493166923523, 1.6877877712249756, 1.6552472114562988, 1.6551495790481567, 1.6600409746170044, 1.6629743576049805, 1.6642475128173828, 1.6662380695343018]
# model_hidden_256_bert_256_lang_32_cast_32_crew_32_batch_8: [2.508727550506592, 2.2758612632751465, 2.1747100353240967, 2.1340363025665283, 2.068742036819458, 2.0449087619781494, 1.9970333576202393, 2.0106470584869385, 2.018690586090088, 1.994417667388916]
# model_leakyReLU_hidden_128_bert_256_lang_16_cast_64_crew_16_batch_32: [1.7625597715377808, 1.6899917125701904, 1.6346031427383423, 1.6450608968734741, 1.6737666130065918, 1.670016884803772, 1.6725292205810547, 1.6842292547225952, 1.6953318119049072]
# model_leakyReLU_hidden_192_bert_192_lang_16_cast_32_crew_16_batch_64: [1.757824182510376, 1.6997736692428589, 1.7094906568527222, 1.6484510898590088, 1.680311918258667, 1.677307367324829, 1.6883188486099243, 1.7052394151687622, 1.7065321207046509, 1.717806100845337]
# model_hidden_256_bert_128_lang_64_cast_64_crew_32_batch_8: [2.0128955841064453, 1.7647507190704346, 1.6930949687957764, 1.687537670135498, 1.6877961158752441, 1.6676483154296875, 1.6738660335540771, 1.6617900133132935, 1.653738021850586, 1.6589730978012085]

# Plot the results on a graph with legend for each hyperparameter combination
# x-axis: epoch
# y-axis: MSE

from matplotlib import pyplot as plt

hidden_128_bert_256_lang_32_cast_32_crew_32_batch_32 = [1.7955729961395264, 1.783509373664856, 1.727493166923523, 1.6877877712249756, 1.6552472114562988, 1.6551495790481567, 1.6600409746170044, 1.6629743576049805, 1.6642475128173828, 1.6662380695343018]
hidden_256_bert_256_lang_32_cast_32_crew_32_batch_8 = [2.508727550506592, 2.2758612632751465, 2.1747100353240967, 2.1340363025665283, 2.068742036819458, 2.0449087619781494, 1.9970333576202393, 2.0106470584869385, 2.018690586090088, 1.994417667388916]
leakyReLU_hidden_128_bert_256_lang_16_cast_64_crew_16_batch_32 = [1.7625597715377808, 1.6899917125701904, 1.6346031427383423, 1.6450608968734741, 1.6737666130065918, 1.670016884803772, 1.6725292205810547, 1.6842292547225952, 1.6953318119049072, 1.685]
leakyReLU_hidden_192_bert_192_lang_16_cast_32_crew_16_batch_64 = [1.757824182510376, 1.6997736692428589, 1.7094906568527222, 1.6484510898590088, 1.680311918258667, 1.677307367324829, 1.6883188486099243, 1.7052394151687622, 1.7065321207046509, 1.717806100845337]
hidden_256_bert_128_lang_64_cast_64_crew_32_batch_8 = [2.0128955841064453, 1.7647507190704346, 1.6930949687957764, 1.687537670135498, 1.6877961158752441, 1.6676483154296875, 1.6738660335540771, 1.6617900133132935, 1.653738021850586, 1.6589730978012085]

epochs = [i for i in range(1, 11)]

plt.plot(epochs, hidden_128_bert_256_lang_32_cast_32_crew_32_batch_32, 'o-', label='(128, 256, 32, 32, 32, 32)', markersize=3)
plt.plot(epochs, hidden_256_bert_256_lang_32_cast_32_crew_32_batch_8, 'o-', label='(256, 256, 32, 32, 32, 8)', markersize=3)
plt.plot(epochs, leakyReLU_hidden_128_bert_256_lang_16_cast_64_crew_16_batch_32, 'o-', label='(128, 256, 16, 64, 16, 32), LeakyReLU', markersize=3)
plt.plot(epochs, leakyReLU_hidden_192_bert_192_lang_16_cast_32_crew_16_batch_64, 'o-', label='(192, 192, 16, 32, 16, 64), LeakyReLU', markersize=3)
plt.plot(epochs, hidden_256_bert_128_lang_64_cast_64_crew_32_batch_8, 'o-', label='(256, 128, 64, 64, 32, 8)', markersize=3)

plt.xlabel('Epoch')
plt.ylabel('MSE for revenue (in 100 millions)')
# plt.title('Loss curves for different hyperparameter combinations')
# Add in the legend that the tuple is (hidden, bert, lang, cast, crew, batch)
plt.legend(title='Notation: (H, B, L, C, D, Batch)', loc='upper right')
plt.show()
