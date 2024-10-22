# F2TNet
[MICCAI'24] F2TNet: FMRI to T1w MRI Knowledge Transfer Network for Brain Multi-phenotype Prediction

Accepted by MICCAI 2024

Using brain imaging data to predict the non-neuroimaging phenotypes at the individual level is a fundamental goal of system neuroscience. Despite its significance, the high acquisition cost of functional Magnetic Resonance Imaging (fMRI) hampers its clinical translation in phenotype prediction, while the analysis based solely on cost-efficient T1-weighted (T1w) MRI yields inferior performance than fMRI. The reasons lie in that existing works ignore two significant challenges. 1) they neglect the knowledge transfer from fMRI to T1w MRI, failing to achieve effective prediction using cost-efficient T1w MRI. 2) They are limited to predicting a single phenotype and cannot capture the intrinsic dependence among various phenotypes, such as strength and endurance, preventing comprehensive and accurate clinical analysis. To tackle these issues, we propose an FMRI to T1w MRI knowledge transfer Network (F2TNet) to achieve cost-efficient and effective analysis on brain multi-phenotype, representing the first attempt in this field, which consists of a Phenotypes-guided Knowledge Transfer (PgKT) module and a modality-aware Multi-phenotype Prediction (MpP) module. Specifically, PgKT aligns brain nodes across modalities by solving a bipartite graph-matching problem, thereby achieving adaptive knowledge transfer from fMRI to T1w MRI through the guidance of multi-phenotype. Then, MpP enriches the phenotype codes with cross-modal complementary information and decomposes these codes to enable accurate multi-phenotype prediction. Experimental results demonstrate that the F2TNet significantly improves the prediction of brain multi-phenotype and outperforms state-of-the-art methods.



If you have any questions, please contact zhibinhe0509@gmail.com.
