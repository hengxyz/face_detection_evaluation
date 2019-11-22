# Compare your discrete ROC curves with other methods
# At terminal: gnuplot discROC.p
set terminal png size 1280, 1200 enhanced font 'Verdana,18'
set size 1,1
set xtics 500
set ytics 0.1
set grid
set ylabel "True positive rate"
set xlabel "False positive"
set xr [0:2000]
set yr [0:1.0]
set key below
set output "discROC.png"
plot  "rocCurves/Face-R-FCN-DiscROC.txt" using 2:1 title 'Face R-FCN' with linespoints pointinterval 50 lw 3 , \
"rocCurves/FaceBoxesDiscROC.txt" using 2:1 title 'FaceBoxes' with linespoints pointinterval 50 lw 3 , \
"rocCurves/SFDDiscROC.txt" using 2:1 title 'SFD' with linespoints pointinterval 50 lw 3 , \
"rocCurves/RSA_disc.txt" using 2:1 title 'RSA' with linespoints pointinterval 50 lw 3 , \
"rocCurves/ScaleFace-DiscROC.txt" using 2:1 title 'ScaleFace' with linespoints pointinterval 50 lw 3 , \
"rocCurves/Face_R-CNN_DiscROC.txt" using 2:1 title 'Face-R-CNN' with linespoints pointinterval 50 lw 3 , \
"rocCurves/HR-ER-DiscROC.txt" using 2:1 title 'HR-ER' with linespoints pointinterval 50 lw 3 , \
"rocCurves/HR-DiscROC.txt" using 2:1 title 'HR' with linespoints pointinterval 50 lw 3 , \
"rocCurves/DeepIRDiscROC(2).txt" using 2:1 title 'DeepIR' with linespoints pointinterval 50 lw 3 , \
"rocCurves/LCDF_DiscROC.txt" using 2:1 title 'LDCF+' with linespoints pointinterval 50 lw 3 , \
"rocCurves/FastCNNDisc.txt" using 2:1 title 'FastCNN' with linespoints pointinterval 50 lw 3 , \
"rocCurves/mxnet-face-fr50-DiscROC.txt" using 2:1 title 'mxnet' with linespoints pointinterval 50 lw 3 , \
"rocCurves/UnitBox-DiscROC.txt" using 2:1 title 'UnitBox' with linespoints pointinterval 80 lw 2 , \
"rocCurves/resnet50-hnm-DiscROC.txt" using 2:1 title 'Xiaomi' with linespoints pointinterval 80 lw 2 , \
"rocCurves/Conv3d_DiscROC.txt" using 2:1 title 'Conv3D' with lines lw 2 , \
"rocCurves/frcnDiscROC.txt" using 2:1 title 'Faster RCNN' with lines lw 2 , \
"rocCurves/Hyperface_DiscROC.txt" using 2:1 title 'Hyperface' with lines lw 2 , \
"rocCurves/MTCNN-DiscROC.txt" using 2:1 title 'Zhang et al.' with lines lw 2 , \
"rocCurves/cvit_DiscROC.txt" using 1:2 title 'Kumar et al.' with lines lw 2 , \
"rocCurves/FDDB_FSA4_2_DiscROC.txt" using 2:1 title 'Barbu et al. 2' with lines lw 2 , \
"rocCurves/DP2MFD_DiscROC.txt" using 2:1 title 'DP2MFD' with lines lw 2 , \
"rocCurves/CCFDiscROC.txt" using 2:1 title 'CCF' with lines lw 2 , \
"rocCurves/Faceness-DiscROC.txt" using 2:1 title 'Faceness' with lines lw 2 , \
"rocCurves/npdfaceDiscROC.txt" using 2:1 title 'NPDFace' with lines lw 2 , \
"rocCurves/MultiresHPM-DiscROC.txt" using 2:1 title 'MultiresHPM' with lines lw 2 , \
"rocCurves/FDDB_FSA4_1_DiscROC.txt" using 2:1 title 'Barbu et al.' with lines lw 2 , \
"rocCurves/DDFD_DiscROC.txt" using 2:1 title 'DDFD' with lines lw 2 , \
"rocCurves/CasCNN-DiscROC.txt" using 2:1 title 'CascadeCNN' with lines lw 2 , \
"rocCurves/jjyan_allROC_DiscROC.txt" using 2:1 title 'Yan et al.' with lines lw 2 , \
"rocCurves/AcfDiscROC.txt" using 2:1 title 'ACF-multiscale' with lines lw 2 , \
"rocCurves/pico-DiscROC.txt" using 2:1 title 'Pico' with lines lw 2 , \
"rocCurves/HeadHunter_DiscROC.txt" using 2:1 title 'HeadHunter' with lines lw 2 , \
"rocCurves/JointCascade_DiscROC.txt" using 2:1 title 'Joint Cascade' with lines lw 2 , \
"rocCurves/BoostedExamplerBased-DiscROC.txt" using 2:1 title 'Boosted Exemplar' with lines lw 2 , \
"rocCurves/SURF_GentleBoost_FTDiscROC.txt" using 2:1 title 'SURF-frontal' with lines lw 2 , \
 "rocCurves/SURF_GentleBoost_MVDiscROC.txt" using 2:1 title 'SURF-multiview' with lines lw 2 , \
 "rocCurves/PEPAdapt_DiscROC.txt" using 2:1 title 'PEP-Adapt' with lines lw 2 , \
 "rocCurves/XZJY_DiscROC.txt" using 2:1 title 'XZJY' with lines lw 2 , \
 "rocCurves/dr40DiscROC.txt" using 2:1 title 'Zhu et al.' with lines lw 2 , \
 "rocCurves/santiDiscROC.txt" using 2:1 title 'Segui et al.' with lines lw 2, \
 "rocCurves/koestingerDiscROC.txt" using 2:1 title 'Koestinger et al.' with lines lw 2, \
 "rocCurves/LiIntelDiscROC.txt" using 2:1 title 'Li et al.' with lines lw 2, \
 "rocCurves/jainDiscROC.txt" using 2:1 title 'Jain et al.' with lines lw 2, \
 "rocCurves/subburamanDiscROC.txt" using 2:1 title 'Subburaman et al.' with lines lw 2, \
 "rocCurves/ViolaJonesScore_n0_DiscROC.txt" using 2:1 title 'Viola-Jones' with lines lw 2, \
 "rocCurves/MikolajczykDiscROC.txt" using 2:1 title 'Mikolajczyk et al.' with lines lw 2, \
 "rocCurves/kienzleDiscROC.txt" using 2:1 title 'Kienzle et al.' with lines lw 2


 
