import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns


def plot_condition_nums_over_time():

    sns.set(font_scale=1.5)

    # plot simple - need DF with x = training step, y = condition number (log scale), hue = norm
    training_steps = 4
    num_norms = 4
    simple_data = {'Training Step': list(range(training_steps)) * num_norms,
                   'Condition Number': [2.9669, 0, 9838589.0, 38.4841, 24.6247, 0, 96763890.0, 147.7037, 27.622883, 0, 28106866.0, 130.34782, 12.277921, 0, 18795718.0, 65.90358],
                   'Norm': ['L2' for __ in range(training_steps)] + ['L1' for __ in range(training_steps)] + ['inf' for __ in range(training_steps)] + ['Frobenius' for __ in range(training_steps)]}
    simple_df = pd.DataFrame.from_dict(simple_data)
    ax = sns.barplot(x='Training Step', y='Condition Number', hue='Norm', data=simple_df)
    ax.set(yscale='log', ylabel='Condition Number')
    ax.set_title('Weight Matrix Condition Number vs. Training Step')
    # for p in ax.patches:
    #     height = p.get_height()
    #     ax.text(p.get_x() + p.get_width()/2., height + 3, '{:1.2e}'.format(height), ha='center')

    plt.show()

    # plot CNN - need DF with x = training step, y = condition number (log scale), hue = norm, stack = layer
    cnn_training_steps = 3
    num_layers = 4
    norm_entries = cnn_training_steps * num_layers
    cnn_training_step_list = list(range(cnn_training_steps)) * num_norms
    norms_list = ['L2'] * cnn_training_steps + ['L1'] * cnn_training_steps + ['inf'] * cnn_training_steps + ['Frobenius'] * cnn_training_steps

    conv0_data = {'Training Step': cnn_training_step_list,
                  'Condition Number': [5.314162, 23.3325, 92.04718, 20.614687, 86.40797, 535.82404, 18.81949, 67.77471, 229.7366, 14.405792, 47.706448, 172.03615],
                  'Norm': norms_list}
    conv0_df = pd.DataFrame.from_dict(conv0_data)

    conv1_data = {'Training Step': cnn_training_step_list,
                  'Condition Number': [7.087775, 138.90384, 2321.0085, 128.61662, 1148.2352, 28681.484, 158.86647, 1395.1996, 44043.207, 103.49358, 555.1632, 7587.9688],
                  'Norm': norms_list}
    conv1_df = pd.DataFrame.from_dict(conv1_data)

    logits_data = {'Training Step': cnn_training_step_list,
                   'Condition Number': [1.7773787, 2104481.2, 22.675634, 12.551755, 5173443.5, 54.22595, 11.616156, 2672598.8, 135.4269, 10.321192, 2151640.5, 38.224712],
                   'Norm': norms_list}
    logits_df = pd.DataFrame.from_dict(logits_data)

    dense_data = {'Training Step': cnn_training_step_list,
                  'Condition Number': [6.09816, 210921500000000000, 260414620.0, 207.77635, 8570022700000.0, 257931090.0, 396.64215, 370268830000000.0, 13741664000.0, 178.53966, 14888170000000.0, 256663840.0],
                  'Norm': norms_list}
    dense_df = pd.DataFrame.from_dict(dense_data)

    layers = ['conv0', 'conv1', 'logits', 'dense']
    dfs = [conv0_df, conv1_df, logits_df, dense_df]
    for i in range(num_layers):
        plt.figure()
        ax = sns.barplot(x='Training Step', y='Condition Number', hue='Norm', data=dfs[i])
        ax.set(yscale='log', ylabel='Condition Number')
        ax.set_title('Weight Matrix Condition Number vs. Training Step for Layer ' + layers[i])
    plt.show()




    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax = sns.barplot(x='Training Step', y='Condition Number', hue='Norm', data=cnn_df)





plot_condition_nums_over_time()