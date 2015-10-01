# change the objectives for defect prediction
ABCD = False

# Mitchell's version space
PDPFPREC = False

PDPF = False
PDPREC = False
PFPREC = False
PD = True
PF = False
PREC = False
GF = False
ACC = False


# change the distribution
# options: UNIFORM, GAUSSIAN
GALE_DISTRIBUTION = "UNIFORM"

# SMOTE the training data
SMOTE = False
SMOTE_BINARY_CLASSIFICATION = True  # consider all the rows as either bugs or not bugs

CULLING = False  # having a culling policy in check pd = 66 and pf = 33
NEW_DE = False   # traditional DE algorithm


IGDMEASURE = False   # USE IGD CALCULATION

