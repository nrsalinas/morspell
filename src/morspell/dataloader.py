import pandas as pd

squ = "../../data/squirrels.csv"

def simpl():
	return "from simpl func"


def squirrel_mat():
	
	cols = [
		'CBL', 'ZB', 'BBC', 'HBC', 'LN', 'WN', 'IOB', 'POB', 'LD', 'LIF',
		'WIF', 'WAIF', 'LAB', 'WAB', 'WAAM', 'LBP', 'PPL', 'LMTR', 'LCP', 'LAP',
		'WPP3', 'WPM1', 'WPM3'
	]
	
	lin = "lineage" # north is a species, the rest of lineages conform the other
	sex = "SEX"
	sample_id = "SampleID"
	dat = pd.read_csv(squ)
	
	dat = dat.rename(
		columns={
			"SampleID": "id",
			"lineage": "lineage",
			"SEX": "sex",
		}
	)
	
	#print(dat.columns)
	#print(dat.groupby("sex").size())

	return (dat[["id", "lineage", "sex"] + cols], cols)

if __name__ == "__main__":

	thmat, morpho_cols = squirrel_mat()
	print(thmat.shape)
	print(morpho_cols)
	print(simpl())

	exit()

