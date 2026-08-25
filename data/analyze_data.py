import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import constants


# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
def print_train_data():
    if not constants.SHOULD_PRINT_DATA_ANALYTICS:
        print('Печать аналитики по данным не выполнена. Флаг SHOULD_PRINT_DATA_ANALYTICS False')
        return

    # вычитали файл в переменную, за индекс взяли колонку PassengerId
    train_data = pd.read_csv("./dataset/train.csv", index_col='PassengerId')

    print("\nвывели первые записи, чтобы визуально оценить таблицу")
    print(train_data.head())

    # 177 Age, 687 Cabin, 2 Embarked - null values
    print("\nинформация о количестве ненулевых признаков в каждой колонке и типе данных")
    print(train_data.info())

    # видим, что в колонке пола нет пропусков и два возможных значения, можно перейти от строчного к числовому представлению
    train_data['Sex'] = transform_sex_to_int(train_data['Sex'])

    # разобьем столбцы по типам
    columns_int64 = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch']
    columns_float64 = ['Age', 'Fare']
    columns_str = ['Name', 'Ticket', 'Cabin', 'Embarked']

    print("\nпо каждому целочисленному столбцу можно посмотреть возможные значения и их частоту")
    for column in columns_int64:
        print(train_data[column].value_counts(normalize=True))
        # те же данные в виде гистограммы
        plt.pie(train_data[column].value_counts(), labels=train_data[column].unique(), autopct='%1.1f%%', startangle=90)
        plt.title(column)
        plt.show()

    print("\nпо каждому вещественному столбцу можно посмотреть возможные значения и их частоту (только на графике, потому что все значения уникальны)")
    for column in columns_float64:
        print(pd.cut(train_data[column], 10).value_counts(normalize=True))
        sns.histplot(train_data[column])
        plt.show()

    print("\nпо графику можем заметить выброс по стоимости билета Fare и подобрать для него значение 95 процентиля")
    print(train_data['Fare'].quantile(0.95))

    print("\nили построим боксплоты для поиска выбросов")
    for column in columns_int64 + columns_float64:
        sns.boxplot(train_data[column])
        plt.show()

    print("\nдля численных типов можно вывести статистики по столбцам (столбцы выбираются автоматически в виде таблицы")
    print(train_data.describe())

    print("\nдля числовых типов можно найти корреляцию")
    print(train_data[columns_int64 + columns_float64].corr())  # значимая корреляция выживания с полом, классом и стоимостью билета. кроме того класс коррелирует со стоимостью билета, число родителей с полом и числом братьев, возраст с классом, братьями и родителями
    # та же корреляция в виде тепловой карты
    sns.heatmap(train_data[columns_int64 + columns_float64].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.show()

    print("\nпостроим график зависимости выживаемости от других числовых признаков. Наблюдения получились интересными для числа супруг/братьев и детей/родителей")
    for column in columns_int64 + columns_float64:
        print(train_data.groupby(column)['Survived'].agg(['count', 'mean', 'median']))
        sns.catplot(x=column, y='Survived', data=train_data, kind="point")
        plt.show()

    # дополнительно нужны графики связи признаков с возрастом (чтобы понять, как лучше заменять пропуски возраста)
    print("\nпостроим график зависимости возраста от других числовых признаков, чтобы понять принцип заполнения пропусков")
    for column in columns_int64 + columns_float64:
        print(train_data.groupby(column)['Age'].agg(['count', 'mean', 'median']))
        sns.catplot(x=column, y='Age', data=train_data, kind="point")
        plt.show()

    # в чужом ноутбуке заметил, что можно смотреть зависимость с категориальными признаками. видно, что тут не случайное распределение
    sns.catplot(x='Embarked', y='Survived', data=train_data, kind="point")
    plt.show()

    # в чужом ноутбуке заметил, что можно смотреть связь статистики по нескольким параметрам. Например, здесь видно, что большинство детей выживает, а стариков нет
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train_data, split=True, ax=ax[0])
    ax[0].set_title('Pclass and Age vs Survived')
    ax[0].set_yticks(range(0, 110, 10))
    sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_data, split=True, ax=ax[1])
    ax[1].set_title('Sex and Age vs Survived')
    ax[1].set_yticks(range(0, 110, 10))
    plt.show()


# Преобразование колонки пола из male, female в 1, 0
def transform_sex_to_int(sex_column: pd.Series) -> pd.Series:
    return (sex_column == 'male').astype(int)
