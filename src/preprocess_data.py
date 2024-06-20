"""Transform raw data to train / val datasets """
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/preprocess_data.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')


IN_FILES = ['data/raw/one_room_data.csv',
            'data/raw/two_room_data.csv',
            'data/raw/three_room_data.csv']

OUT_TRAIN = 'data/proc/train.csv'
OUT_VAL = 'data/proc/val.csv'

TRAIN_SIZE = 0.85


def main(args):
    main_dataframe = pd.read_csv(args.input[0], sep=';')
    for i in range(1, len(args.input)):
        data = pd.read_csv(args.input[i], sep=';')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    # new_dataframe = main_dataframe[['url_id', 'total_meters', 'price']].set_index('url_id')
    main_dataframe['first_floor'] = (main_dataframe['floor'] == 1) * 1
    main_dataframe['last_floor'] = (main_dataframe['floor'] == main_dataframe['floors_count']) * 1
    main_dataframe = main_dataframe[['rooms_count', 'author_type', 'floor', 'floors_count', 'first_floor', 'last_floor', 'total_meters', 'district', 'price']]
    main_dataframe = main_dataframe.dropna()

    # border = int(args.split * len(new_df))
    # train_df, val_df = new_df[0:border], new_df[border:-1]
    test_size = 0.15
    X_train, X_test = train_test_split(main_dataframe, test_size = test_size, random_state = 42)
    X_train.to_csv(OUT_TRAIN)
    X_test.to_csv(OUT_VAL)
    logger.info(f'Write {args.input} to train.csv and val.csv. Train set size: {1-test_size}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=float, 
                        help='Split test size',
                        default=TRAIN_SIZE)
    parser.add_argument('-i', '--input', nargs='+',
                        help='List of input files', 
                        default=IN_FILES)
    args = parser.parse_args()
    main(args)
