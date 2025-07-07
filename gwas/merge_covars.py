import os.path

import pandas as pd


def main(args):
    pheno = args.pheno
    pca = args.pca
    out = args.out
    data_path = args.data_path

    if not os.path.exists(os.path.join('data', data_path, pca)):
        print("PCA results not found. Please run PCA first.")
        return
    # pca results
    pca_results = pd.read_csv(os.path.join('data', data_path, pca), sep=' ', header=None)
    pca_results.columns = ['FID', 'IID'] + ['PC' + str(i) for i in range(1, 11)]

    # phenotype
    df_pheno = pd.read_csv(f'data/{data_path}/pheno_data.txt', sep=' ')

    # merge
    df = pd.merge(df_pheno, pca_results, on=['FID', 'IID'])

    # save
    df.to_csv(os.path.join('data', data_path, out), sep=' ', index=False, na_rep='NA')

    print(df[pheno].value_counts())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='insomnia_score', help='phenotype name')
    parser.add_argument('--pheno', type=str, default='pheno', help='phenotype name')
    parser.add_argument('--pca', type=str, default='pca.eigenvec', help='path to pca results')
    parser.add_argument('--out', type=str, default='pheno_data_pca.txt', help='path to output file')
    args = parser.parse_args()
    main(args)
