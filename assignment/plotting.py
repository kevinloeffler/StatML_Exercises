import matplotlib.pyplot as plt

from assignment.statistics import ModelPerformance


def plot_model_categories(performances: list[tuple[list[str], ModelPerformance]], by: str):
    def compare(first, second):
        if by == 'r2':
            return getattr(first[1], by) > getattr(second[1], by)
        else:
            return getattr(first[1], by) < getattr(second[1], by)

    models = [None] * 8
    for performance in performances:
        models_index = len(performance[0]) - 1
        if models[models_index] is not None and compare(performance, models[models_index]):
            continue
        else:
            models[models_index] = performance

    fig, ax = plt.subplots()
    labels = [num for num in range(1, len(models) + 1)]
    perfs = [getattr(model[1], by) for model in models]

    ax.set_title(f'{by} scores')
    ax.set_xlabel('best model with # of sensors')
    ax.set_yscale('log')
    ax.bar(labels, perfs)
    plt.show()
