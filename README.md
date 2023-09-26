# DGATCCDA
DeepWalk-aware graph attention networks with CNN for circRNA-drug sensitivity association identification

## Code
### Environment Requirement
The code has been tested running under Python 3.8.0. The required packages are as follows
- torch_geometric==2.3.0
- pandas == 1.5.3
- numpy == 1.23.5
- scipy == 1.10.1
- torch == 1.13.1+cu116

## DataSets
1.the association represents the association matrix of circRNA and drug
2.gene_seq_sim represents the similarity in sequence of circRNAs host genes
3.drug_str_sim represents Structural similarity of drugs

## code
main.py: the entrance of the program;
