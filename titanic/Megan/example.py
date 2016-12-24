# From: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

import pandas as pd
import seaborn as sns

from process_data import clean_data, get_deck, get_family_size, \
                        get_person_title, get_person_surname, \
                        merge_rare_title
from utility import load_data, preview_data, set_print_style, filter_data
from visualization import show_embarked_pclass_fare, show_family_survival, \
                            show_title_by_sex, show_surname_amount


# load data
train, test = load_data()
preview_data([train, test])
set_print_style()

# Show title counts by sex_dummies
surname_tf = get_person_title(train)
show_title_by_sex(surname_tf)

surname_tf = merge_rare_title(surname_tf)
show_title_by_sex(surname_tf)

# Finally, grab surname from passenger name
surname_tf = get_person_surname(train)
show_surname_amount(surname_tf)
#TODO: how to get number of types when value_counts output long list?


# Show survival of family size
get_family_size(train)
show_family_survival(train)
#TODO: why there is an empty figure?

# Get deck from Cabin
train = get_deck(train)

# Find out empty embarked passenger and fill the value by fare
copy = filter_data(train, 'PassengerId', [62, 830])
show_embarked_pclass_fare(copy)
train = clean_data(train)

print(train.head(10))

# sns.plt.show()
