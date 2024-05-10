### Run the example cases

```
# patch_v2 branch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240306_111344_480307_STCNNT_MRI_NN_120.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240306_111344_480307_STCNNT_MRI_NN_120.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1_epoch-20.pth

RES_DIR=res_1st_hrnet_TLGTLG_TLGTLG_perf_cine_NN120


model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1_20231204_194332_632800_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1_20231204_194332_632800_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1T1L1G1_T1L1G1T1L1G1_T1L1G1_epoch-50.pth

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231207_032002_166088_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/last_checkpoint

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231210_180813_326625_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_5

RES_DIR=res_1st_hrnet_TLGTLG_TLGTLG_TLG_NN100

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231210_180813_326625_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_NN50_100_vgg10_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231212_161857_681526_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_9

model=/isilon/lab-xue/projects/data/logs/mri-1st_NN40_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231211_082619_350342_STCNNT_MRI_NN_40.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/mri-1st_NN40_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231211_082619_350342_STCNNT_MRI_NN_40.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1_epoch-30.pth

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_010220_285163_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_1

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_010247_137822_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_vgg10_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_011526_172263_STCNNT_MRI_NN_80.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/last_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1_epoch-30.pth

RES_DIR=res_1st_hrnet_TLGTLG_TLGTLG_more_epochs_NN100

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_UNET_T1L1G1_T1L1G1_20231225_190319_401166_STCNNT_MRI_NN_100.0_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri-1st_STCNNT_UNET_T1L1G1_T1L1G1_20231225_190319_401166_STCNNT_MRI_NN_100.0_C-32-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-30.pth

RES_DIR=res_1st_unet_TLG_TLG

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_STCNNT_UNET_T1T1T1_T1T1T1_T1T1T1_20231223_170737_136110_STCNNT_MRI_NN_100.0_C-32-1_amp-False_complex_residual_weighted_loss_snr-T1T1T1_T1T1T1_T1T1T1/mri-1st_more_epochs_STCNNT_UNET_T1T1T1_T1T1T1_T1T1T1_20231223_170737_136110_STCNNT_MRI_NN_100.0_C-32-1_amp-False_complex_residual_weighted_loss_snr-T1T1T1_T1T1T1_T1T1T1_epoch-30.pth

RES_DIR=res_1st_unet_TTT_TTT

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_UNET_C3C3C3_C3C3C3_20231215_212141_363603_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3/mri-1st_STCNNT_UNET_C3C3C3_C3C3C3_20231215_212141_363603_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3_epoch-30.pth
RES_DIR=res_unet_C3C3C3_C3C3C3

model_type_str=STCNNT_MRI
scaling_factor=1.0

model=/isilon/lab-xue/projects/data/logs/mri-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231217_212646_307157_MRI_double_net_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/checkpoint_epoch_4
model=/isilon/lab-xue/projects/data/logs/mri-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231217_212646_307157_MRI_double_net_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/mri-2nd_STCNNT_HRNET_T1L1G1_T1L1G1_20231217_212646_307157_MRI_double_net_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1_epoch-30.pth

model_type_str=MRI_double_net
scaling_factor=1.0

RES_DIR=res_2nd_hrnet_TLGTLG_TLGTLG_more_epochs_super_resolution

export CUDA_VISIBLE_DEVICES=0
export DISABLE_FLOAT16_INFERENCE=True

# for the scale by signal test

model=/isilon/lab-xue/projects/data/logs/mri-1st_scale_by_signal_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240428_212543_009291_STCNNT_MRI_NN_120.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_59

RES_DIR=res_1st_hrnet_TLG_TLGTLG_scale_by_signal

# ======================================================================
# quick test case

scaling_factor=1.0

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231102_HV/meas_MID00542_FID20263_G25_4CH_CINE_256_R3ipat_85phase_res_BH/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231102_HV/meas_MID00542_FID20263_G25_4CH_CINE_256_R3ipat_85phase_res_BH/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

scaling_factor=2

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230822/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_2015269043_2015269052_266_20230822-134501/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230822/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_2015269043_2015269052_266_20230822-134501/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230907/Perfusion_AIF_42110_68970660_68970669_718_20230907-110614/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230907/Perfusion_AIF_42110_68970660_68970669_718_20230907-110614/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

scaling_factor=3

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00169_FID22479_G33_Perfusion_trufi_sr_tpat_3_192res_TI120_BW450/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00169_FID22479_G33_Perfusion_trufi_sr_tpat_3_192res_TI120_BW450/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00173_FID22483_G33_Perfusion_trufi_sr_tpat_4_256res_TI120_BW450/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00173_FID22483_G33_Perfusion_trufi_sr_tpat_4_256res_TI120_BW450/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00182_FID22492_G33_4CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00182_FID22492_G33_4CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00183_FID22493_G33_3CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00183_FID22493_G33_3CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# high res perf

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_41837_2466125178_2466125187_673_20240109-140452/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_41837_2466125178_2466125187_673_20240109-140452/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_42110_414467838_414467847_484_20240109-160845/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_42110_414467838_414467847_484_20240109-160845/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024_normal_res/20240116/Perfusion_AIF_Q_mapping_42110_441437177_441437186_821_20240116-145043/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024_normal_res/20240116/Perfusion_AIF_Q_mapping_42110_441437177_441437186_821_20240116-145043/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024_normal_res/20240116/Perfusion_AIF_Q_mapping_42110_441437177_441437186_822_20240116-145201/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024_normal_res/20240116/Perfusion_AIF_Q_mapping_42110_441437177_441437186_822_20240116-145201/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240113/Perfusion_AIF_Q_mapping_42110_434652876_434652885_351_20240113-132619/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240113/Perfusion_AIF_Q_mapping_42110_434652876_434652885_351_20240113-132619/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240118/Perfusion_AIF_Q_mapping_42110_04434847_04434852_1740_20240118-100851/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240118/Perfusion_AIF_Q_mapping_42110_04434847_04434852_1740_20240118-100851/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230904/Perfusion_AIF_STCNNT_41837_2049151221_2049151230_432_20230904-170739/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230904/Perfusion_AIF_STCNNT_41837_2049151221_2049151230_432_20230904-170739/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# G33 perf

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20240118_Contrast_HV/meas_MID00179_FID25685_Adeno_stress_G33_Perf_2RR_SAX4_4_2_CHA_R3_192res_TI135/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20240118_Contrast_HV/meas_MID00179_FID25685_Adeno_stress_G33_Perf_2RR_SAX4_4_2_CHA_R3_192res_TI135/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20240118_Contrast_HV/meas_MID00180_FID25686_rest_G33_Perf_2RR_SAX4_4_2_CHA_R3_192res_TI135/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20240118_Contrast_HV/meas_MID00180_FID25686_rest_G33_Perf_2RR_SAX4_4_2_CHA_R3_192res_TI135/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}


# G33, LGE

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00188_FID22498_G33_SAX5_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00188_FID22498_G33_SAX5_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# ======================================================================
# generalization test

model_type_str=STCNNT_MRI
scaling_factor=1.0

export CUDA_VISIBLE_DEVICES=1
export DISABLE_FLOAT16_INFERENCE=True

# -----------------------------

model=/isilon/lab-xue/projects/data/logs/mri-1st_scale_by_signal_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240428_212543_009291_STCNNT_MRI_NN_120.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_59
RES_DIR=res_1st_hrnet_TLG_TLGTLG_scale_by_signal

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_41837_2466125178_2466125187_673_20240109-140452/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_41837_2466125178_2466125187_673_20240109-140452/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str} --scale_by_signal

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00182_FID22492_G33_4CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00182_FID22492_G33_4CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str} --scale_by_signal

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_42110_414467838_414467847_484_20240109-160845/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_42110_414467838_414467847_484_20240109-160845/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str} --scale_by_signal

# -----------------------------

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240419_095254_046420_STCNNT_MRI_NN_120.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/checkpoint_epoch_119
model=/isilon/lab-xue/projects/data/logs/mri-1st_HRNET_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20240420_131451_309670_STCNNT_MRI_NN_120.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_53
RES_DIR=res_1st_hrnet_TLG_TLGTLG

scaling_factor=1

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_41837_2466125178_2466125187_673_20240109-140452/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_41837_2466125178_2466125187_673_20240109-140452/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00182_FID22492_G33_4CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231205_Contrast_PT/meas_MID00182_FID22492_G33_4CH_FB_de_tpat3_res256_Ave16_BW610_PHASres84/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_42110_414467838_414467847_484_20240109-160845/DebugOutput/ --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_2024/20240109/Perfusion_AIF_Q_mapping_42110_414467838_414467847_484_20240109-160845/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# ======================================================================
# snr level test

case_dir=Retro_Lin_Cine_2DT_LAX_GLS_66016_078855422_078855431_409_20230613-154734_slc_1

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/projects/mri/data/mri_test/${case_dir}/ --output_dir /isilon/lab-xue/projects/mri/data/mri_test/${case_dir}/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname noisy --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# ======================================================================
# val and test data

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1_20231206_221309_194177_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/final_epoch
model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1_20231207_144811_706305_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1_T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231209_024351_542954_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/final_epoch
model=/isilon/lab-xue/projects/data/logs/mri-1st_weighted_loss_snr_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231210_182221_802807_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231207_032002_166088_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch
model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231209_040057_261280_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231210_180813_326625_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_more_epochs_NN50_100_vgg10_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231212_161857_681526_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/best_checkpoint_epoch_0

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_C3C3C3_C3C3C3_20231209_031038_755604_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C3C3C3_C3C3C3/last_epoch


model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1T1T1_T1T1T1_20231210_182551_453679_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1T1T1_T1T1T1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1T1T1_T1T1T1T1T1T1_20231209_044111_071741_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1T1T1_T1T1T1T1T1T1/final_epoch
model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_T1T1T1_T1T1T1T1T1T1_20231210_220041_001051_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1T1T1_T1T1T1T1T1T1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_UNET_T1L1G1_T1L1G1_20231209_031349_460210_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/final_epoch
model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_UNET_T1T1T1_T1T1T1_20231211_151112_817411_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1T1T1_T1T1T1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_UNET_T1L1G1_T1L1G1T1L1G1_20231211_151025_373473_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_NN80_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231211_144149_648528_STCNNT_MRI_NN_80.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_NN40_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231211_082619_350342_STCNNT_MRI_NN_40.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_UNET_T1T1T1_T1T1T1T1T1T1_20231212_131941_140322_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1T1T1_T1T1T1T1T1T1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_only_white_noise_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231211_171206_344686_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_only_white_noise-T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_weighted_loss_snr_STCNNT_HRNET_T1T1T1_T1T1T1_20231212_163317_044928_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1T1T1_T1T1T1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_no_gmap_STCNNT_HRNET_T1L1G1_T1L1G1T1L1G1_20231211_190043_650972_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_ignore_gmap-T1L1G1_T1L1G1T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_STCNNT_HRNET_C2C2C2_C2C2C2_20231214_003139_652316_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual-C2C2C2_C2C2C2/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_vgg10_30_more_epochs_STCNNT_HRNET_T1L1G1T1L1G1_T1L1G1T1L1G1_20231214_200159_581006_STCNNT_MRI_NN_100.0_C-64-1_amp-False_complex_residual_weighted_loss_snr-T1L1G1T1L1G1_T1L1G1T1L1G1/latest_epoch

test_sets="test_2D_sig_2_80_500.h5 test_2DT_sig_2_80_1000.h5"
test_sets="test_2DT_sig_2_80_2000.h5"

model_type_str=STCNNT_MRI

torchrun --standalone --nproc_per_node 4 ./projects/mri/run.py --ddp --data_dir /isilon/lab-xue/projects/mri/data --log_dir /isilon/lab-xue/projects/data/logs --complex_i --train_model False --continued_training True --project mri --prefetch_factor 8 --batch_size 16 --time 12 --num_uploaded 128 --ratio 50 50 100 --max_load -1 --model_type ${model_type_str} --train_files BARTS_RetroCine_3T_2023.h5 --test_files ${test_sets} --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_data_types 2dt 2dt 2d 2dt --backbone_model STCNNT_HRNET --wandb_dir /isilon/lab-xue/projects/mri/wandb --override --pre_model_load_path ${model}_pre.pth --backbone_model_load_path ${model}_backbone.pth --post_model_load_path ${model}_post.pth --post_model_of_1st_net ${model}_post.pth --freeze_pre True --freeze_backbone True --disable_LSUV --post_backbone STCNNT_HRNET --post_hrnet.block_str T1L1G1 T1L1G1 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 2.0 --max_noise_level 80.0 --mri_height 32 64 --mri_width 32 64 --run_name Test_${model} --run_notes Test_${model} --n_head 64 --eval_train_set False --eval_val_set False --eval_test_set True

model=/isilon/lab-xue/projects/data/logs/mri-1st_lr1e-4_omnivore_T1L1G1_T1L1G1_20231211_150148_002152_omnivore_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/final_epoch
model=/isilon/lab-xue/projects/data/logs/mri-1st_lr1e-3_plateau_omnivore_T1L1G1_T1L1G1_20231211_133403_995287_omnivore_MRI_NN_100.0_C-64-1_amp-False_complex_residual-T1L1G1_T1L1G1/final_epoch

model=/isilon/lab-xue/projects/data/logs/mri-1st_lr1e-4_omnivore_T1L1G1_T1L1G1_20231210_160530_129063_omnivore_MRI_NN_100.0_C-64-1_amp-True_complex_residual-T1L1G1_T1L1G1/final_epoch

test_sets="test_2DT_sig_2_80_2000.h5"

torchrun --standalone --nproc_per_node 4 ./projects/mri/run.py --ddp --data_dir /isilon/lab-xue/projects/mri/data --log_dir /isilon/lab-xue/projects/data/logs --complex_i --train_model False --continued_training True --project mri --prefetch_factor 8 --batch_size 16 --time 12 --num_uploaded 128 --ratio 50 50 100 --max_load -1 --model_type omnivore_MRI --train_files BARTS_RetroCine_3T_2023.h5 --test_files ${test_sets} --train_data_types 2dt 2dt 2dt 2dt 2dt 2dt 2dt 2dt 3d --test_data_types 2dt 2dt 2d 2dt --backbone_model omnivore --wandb_dir /isilon/lab-xue/projects/mri/wandb --override --pre_model_load_path ${model}_pre.pth --backbone_model_load_path ${model}_backbone.pth --post_model_load_path ${model}_post.pth --post_model_of_1st_net ${model}_post.pth --freeze_pre True --freeze_backbone True --disable_LSUV --post_backbone STCNNT_HRNET --post_hrnet.block_str T1L1G1 T1L1G1 --losses mse perpendicular perceptual charbonnier gaussian3D --loss_weights 1.0 1.0 1.0 1.0 1.0 1.0 --min_noise_level 2.0 --max_noise_level 80.0 --mri_height 32 64 --mri_width 32 64 --run_name Test_${model} --run_notes Test_${model} --n_head 64 --eval_train_set False --eval_val_set False --eval_test_set True

# ======================================================================

python3 ./mri/eval_mri.py --test_files /isilon/lab-xue/projects/mri/data/retro_cine_3T_sigma_1_20_repeated_test.h5 --saved_model_path ${model} --num_uploaded 256 --save_samples --num_saved_samples 1024 --results_path /isilon/lab-xue/projects/mri/results/${RES_DIR} --model_type ${model_type_str} --scaling_factor 1.0

python3 ./mri/eval_mri.py --test_files /isilon/lab-xue/projects/mri/data/retro_cine_3T_sigma_1_20_repeated_test_3rd.h5 --saved_model_path ${model} --num_uploaded 256 --save_samples --num_saved_samples 1024 --results_path /isilon/lab-xue/projects/mri/results/${RES_DIR} --model_type ${model_type_str} --scaling_factor 1.0

# ======================================================================
## Run the batch

### Run the WB LGE

```
# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --saved_model_path $model  --model_type ${model_type_str}

# on the moco+ave images

python3 ./mri/run_inference_batch.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_ave --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

```

### Run the DB LGE
```
# on the raw images
python3 ./mri/run_inference_batch.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising_AI_on_raw --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --saved_model_path $model --model_type ${model_type_str}

# on the moco+ave images

python3 ./mri/run_inference_batch.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_DB_LGE_Denoising_2023_AI_denoising_AI_on_ave --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

```

### Perfusion
```

# 3T
python3 ./mri/run_inference_batch.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

# 1.5T
python3 ./mri/run_inference_batch.py --input_dir /isilon/lab-kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising --output_dir /isilon/lab-kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str} # --num_batches_to_process 2

```

# ======================================================================
# local PSF test

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/1280/ --output_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/1280//${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname test_1280_epoch_-1_1280_sigma_1.00_x --gmap_fname test_1280_epoch_-1_1280_sigma_1.00_gmap --saved_model_path $model --model_type ${model_type_str}


python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/400/ --output_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/test_-1/400//${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname test_400_epoch_-1_400_sigma_1.00_x --gmap_fname test_400_epoch_-1_400_sigma_1.00_gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/ori/ --output_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/res/ori --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname x --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/perturb/ --output_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/res/perturb --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname x --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/perturb_2x/ --output_dir /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/retro_cine_cases_for_quantification/1280/LPSF/bp/res/perturb_2x --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname x --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# ======================================================================

# knee

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/DataForHui/kneeH5/imagedata/20190104_200942_meas_MID00034_FID06440_t2_tse_tra/res/DebugOutput/ --output_dir /isilon/lab-kellman/Share/data/DataForHui/kneeH5/imagedata/20190104_200942_meas_MID00034_FID06440_t2_tse_tra/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input_ave0 --gmap_fname gmap_ave0 --saved_model_path $model --model_type ${model_type_str}

# spine

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/DataForHui/spineH5/imagedata/20181207_181828_meas_MID00150_FID01036_t2_tse_sag_p2/res/DebugOutput/ --output_dir /isilon/lab-kellman/Share/data/DataForHui/spineH5/imagedata/20181207_181828_meas_MID00150_FID01036_t2_tse_sag_p2/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input_ave0 --gmap_fname gmap_ave0 --saved_model_path $model --model_type ${model_type_str}

# neuro
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/neuro/meas_MID00083_FID14721_t1_mprage_1mm_p4_pos50_ACPC_check/ --output_dir /isilon/lab-kellman/Share/data/neuro/meas_MID00083_FID14721_t1_mprage_1mm_p4_pos50_ACPC_check/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname im --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/neuro/meas_MID00094_FID14732_t2_spc_sag_1mm_p2X2/res/DebugOutput/ --output_dir /isilon/lab-kellman/Share/data/neuro/meas_MID00094_FID14732_t2_spc_sag_1mm_p2X2/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

#lung
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/DebugOutput/ --output_dir /isilon/lab-kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model --model_type ${model_type_str}

#lung
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/DebugOutput/ --output_dir /isilon/lab-kellman/Share/data/LungTSE_rawData/20181218_124140_meas_MID00542_FID03365_t2_tse_tra_p2_320_trig/res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model --model_type ${model_type_str}


T:/Share/data/neuro/meas_MID00083_FID14721_t1_mprage_1mm_p4_pos50_ACPC_check

T:/Share/data/LungTSE_rawData

T:/Share/data/DataForHui/kneeH5/imagedata/20190104_200259_meas_MID00033_FID06439_pd_tse_sag_384

T:/Share/data/DataForHui/spineH5/imagedata/20181207_181828_meas_MID00150_FID01036_t2_tse_sag_p2


# ======================================================================

# WB LGE


python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114 --output_dir /isilon/lab-kellman/ReconResults/denoising/BWH/BWH_WB_LGE_2023_AI_denoising/20230208/WB_LGE_MOCO_AVE_OnTheFly_41144_01418721_01418731_1929_20230208-164114/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname raw_im --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1525437056_1525437065_1150_20230405-115358 --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_WB_LGE_Denoising_2023_AI_denoising_AI_on_raw/WB_LGE_MOCO_AVE_OnTheFly_41837_1525437056_1525437065_1150_20230405-115358/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_178_20230904-105747/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_178_20230904-105747/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_179_20230904-105942/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT_LGE/20230904/WB_LGE_MOCO_AVE_STCNNT_41837_2049151069_2049151078_179_20230904-105942/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gfactor --saved_model_path $model --model_type ${model_type_str}

# -------------------------------------------------------
# 3T perfusion

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715 --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210316/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_2173099_2173108_894_20210316-154715/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210721/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_11661307_11661316_1331_20210721-151837 --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2021_AI_denoising/20210721/Perfusion_AIF_TwoEchoes_Interleaved_R2_66016_11661307_11661316_1331_20210721-151837/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}


# -------------------------------------------------------
# 1.5T perfusion

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986417889_1986417898_401_20230815-094846/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986417889_1986417898_401_20230815-094846/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986418053_1986418062_697_20230815-154903/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230815/Perfusion_AIF_TwoEchoes_Interleaved_R2_41837_1986418053_1986418062_697_20230815-154903/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230903/Perfusion_AIF_STCNNT_42110_56257534_56257543_3000004_20230903-172928/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230903/Perfusion_AIF_STCNNT_42110_56257534_56257543_3000004_20230903-172928/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising/20221005/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636 --output_dir /isilon/lab-kellman/ReconResults/denoising/Barts_Perf_Denoising_2022_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_121388454_121388463_784_20221005-121636/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_0343475_0343484_1759_20210118-083835 --output_dir /isilon/lab-kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_0343475_0343484_1759_20210118-083835/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_5346528_5346537_161_20210105-104117 --output_dir /isilon/lab-kellman/ReconResults/denoising/1p5T_for_testing/Barts_Perf_Denoising_2021_AI_denoising_AI/Perfusion_AIF_TwoEchoes_Interleaved_R2_42110_5346528_5346537_161_20210105-104117/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508 --output_dir /isilon/lab-kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230424/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_21768048_21768055_241_20230424-122508/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638 --output_dir /isilon/lab-kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230119/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_51358165_51358175_441_20230119-155638/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441 --output_dir /isilon/lab-kellman/ReconResults/denoising/MINNESOTA_UHVC/MINNESOTA_UHVC_Perfusion_2023_AI_denoising/20230117/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_46001532_46001542_263_20230117-110441/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/MINNESOTA_UHVC_perf_stcnnt/20220517/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_26210628_26210638_148_20220517-100037/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/MINNESOTA_UHVC_perf_stcnnt/20220517/Perfusion_AIF_TwoEchoes_Interleaved_R2_169958_26210628_26210638_148_20220517-100037/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

# high res perfusion

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/Perfusion_AIF_2E_NL_Cloud_42170_49443333_49443342_657_20190330-124527/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/Perfusion_AIF_2E_NL_Cloud_42170_49443333_49443342_657_20190330-124527/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_072714971_072714980_550_20180718-175707/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_072714971_072714980_550_20180718-175707/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_303_20180808-183735/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_303_20180808-183735/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_322_20180808-185431/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_322_20180808-185431/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_410_20180906-125352/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_410_20180906-125352/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_49443137_49443146_219_20190329-133337/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/Perfusion_AIF_2E_NL_Cloud_42170_49443137_49443146_219_20190329-133337/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230905/Perfusion_AIF_41837_2049151441_2049151450_882_20230905-145010/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230905/Perfusion_AIF_41837_2049151441_2049151450_882_20230905-145010/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230905/Perfusion_AIF_Q_mapping_41837_2049151252_2049151261_500_20230905-085551/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_Perfusion_high_res_2023/20230905/Perfusion_AIF_Q_mapping_41837_2049151252_2049151261_500_20230905-085551/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}


cases=(
        Perfusion_AIF_2E_NL_Cloud_42170_072714971_072714980_550_20180718-175707
        Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_303_20180808-183735
        Perfusion_AIF_2E_NL_Cloud_42170_123211405_123211414_322_20180808-185431
        Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_408_20180906-123716
        Perfusion_AIF_2E_NL_Cloud_42170_190304911_190304920_410_20180906-125352
        Perfusion_AIF_2E_NL_Cloud_42170_49443137_49443146_219_20190329-133337
        Perfusion_AIF_2E_NL_Cloud_42170_49443461_49443470_1001_20190331-092527
        Perfusion_AIF_2E_NL_Cloud_42170_49443486_49443495_1068_20190331-102251
        Perfusion_AIF_2E_NL_Cloud_42170_55882022_55882031_89_20190401-084425
        Perfusion_AIF_2E_NL_Cloud_42170_90141277_90141286_902_20170622-150439
        Perfusion_AIF_2E_NL_Cloud_42170_90141277_90141286_916_20170622-151616
        Perfusion_AIF_2E_NL_Cloud_42170_99913385_99913394_383_20171129-160152
        Perfusion_AIF_2E_NL_Cloud_66097_19853195_19853203_3000002_20171207-151256
        Perfusion_AIF_2E_NL_Cloud_66097_29373222_29373230_45_20180926-092303
        Perfusion_AIF_2E_NL_Cloud_66097_46576496_46576504_228_20181010-161347
        Perfusion_AIF_2E_NL_Cloud_66097_52964208_52964216_3000002_20180110-161515
        Perfusion_AIF_2E_NL_Cloud_66097_5709937_5709942_106_20181016-173229
    )

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"

    python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/${cases[$index]}/cloud_flow_res/DebugOutput --output_dir /isilon/lab-kellman/Share/data/perfusion/cloud/cloud_ai/${cases[$index]}/cloud_flow_res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str} --patch_size_inference 64

done

cases=(
    Perfusion_AIF_2E_NL_Cloud_66097_29373222_29373230_45_20180926-092303
        Perfusion_AIF_2E_NL_Cloud_66097_5709937_5709942_106_20181016-173229
    )

# -------------------------------------------------------

# R4
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804 --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_528_20230616-173804/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}

# R5
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849 --output_dir /isilon/lab-kellman/ReconResults/denoising/BARTS/BARTS_RTCine_AI_2023_AI_denoising/20230616/RT_Cine_LIN_41837_1769771291_1769771300_529_20230616-173849/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 10.0 --gmap_scaling 100.0 --input_fname im --saved_model_path $model --model_type ${model_type_str}


# -------------------------------------------------------
# free max cine

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00156_FID07562_G25_2CH_CINE_256_R2/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00163_FID07569_G25_4CH_CINE_256_R4/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00685_FID08073_G25_4CH_CINE_256_R4ipat/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230710_NV_AI/meas_MID00687_FID08077_REPEAT_G25_4CH_CINE_256_R4ipat/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00417_FID09075_G25_4CH_CINE_192_R4ipat_BW401/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00417_FID09075_G25_4CH_CINE_192_R4ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00418_FID09076_G25_4CH_CINE_192_R3ipat_BW401/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00418_FID09076_G25_4CH_CINE_192_R3ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00419_FID09077_G25_4CH_CINE_192_R2ipat_BW401/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00419_FID09077_G25_4CH_CINE_192_R2ipat_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00412_FID09070_G25_4CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00412_FID09070_G25_4CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00413_FID09071_G25_3CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00413_FID09071_G25_3CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00414_FID09072_G25_2CH_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00414_FID09072_G25_2CH_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00416_FID09074_G25_SAX_CINE_256_R4ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00416_FID09074_G25_SAX_CINE_256_R4ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00411_FID09069_G25_SAX_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00411_FID09069_G25_SAX_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00410_FID09068_REPEAT_FOV360_G25_3CH_CINE_256_R2ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00410_FID09068_REPEAT_FOV360_G25_3CH_CINE_256_R2ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00407_FID09065_G25_2CH_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00407_FID09065_G25_2CH_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00406_FID09064_G25_4CH_CINE_256_R3ipat_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00406_FID09064_G25_4CH_CINE_256_R3ipat_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}


python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231005_HV1/meas_MID00163_FID16270_G25_CH4_CINE_256_R3ipat_85phase_res_BH/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20231005_HV1/meas_MID00163_FID16270_G25_CH4_CINE_256_R3ipat_85phase_res_BH/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# free max perf

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00175_FID07581_G25_Perfusion_trufi_sr_tpat_3_192res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_0 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_1 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00186_FID04084_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_2 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input2 --gmap_fname gmap2 --saved_model_path $model  --model_type ${model_type_str}


python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_0 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_1 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input1 --gmap_fname gmap1 --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230517_Pat_MI/meas_MID00225_FID04597_REST_G33_perfusion_trufi_sr_Tpat_3_192/${RES_DIR}_2 --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input2 --gmap_fname gmap2 --saved_model_path $model  --model_type ${model_type_str}

# free max perf 256
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00176_FID07582_G25_Perfusion_trufi_sr_tpat_4_256res/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# new NV
python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00421_FID09079_G25_Perfusion_trufi_sr_tpat_3_192res_BW401/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00421_FID09079_G25_Perfusion_trufi_sr_tpat_3_192res_BW401/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00422_FID09080_G25_Perfusion_trufi_sr_tpat_4_256res_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00422_FID09080_G25_Perfusion_trufi_sr_tpat_4_256res_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# free max LGE

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00092_FID08362_G25_4CH_FB_de_tpat4_res256_Ave16/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00092_FID08362_G25_4CH_FB_de_tpat4_res256_Ave16/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00091_FID08361_G25_2CH_FB_de_tpat3_res256_Ave16/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230712_NV_AI/meas_MID00091_FID08361_G25_2CH_FB_de_tpat3_res256_Ave16/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00211_FID04109_G25_4CH_FB_de_snapshot_p3_BW500/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00211_FID04109_G25_4CH_FB_de_snapshot_p3_BW500/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00215_FID04113_G25_SAX_FB_de_snapshot_p3_BW500/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230511_Pat_MI_Cardiac_Gd/meas_MID00215_FID04113_G25_SAX_FB_de_snapshot_p3_BW500/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00429_FID09087_G25_SAX3_FB_de_tpat3_res256_Ave16_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00429_FID09087_G25_SAX3_FB_de_tpat3_res256_Ave16_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00430_FID09088_G25_4CH_FB_de_tpat4_BW450_res256_Ave24_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00430_FID09088_G25_4CH_FB_de_tpat4_BW450_res256_Ave24_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00431_FID09089_G25_SAX3_FB_de_tpat4_BW450_res256_Ave24_BW399/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230720_NV/meas_MID00431_FID09089_G25_SAX3_FB_de_tpat4_BW450_res256_Ave24_BW399/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model  --model_type ${model_type_str}

# -------------------------------------------------------
# high-res cmr

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/Perfusion_AIF_STCNNT_41837_2020136443_2020136452_811_20230824-122633/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/Perfusion_AIF_STCNNT_41837_2020136443_2020136452_811_20230824-122633/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136389_2020136398_731_20230824-111014/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136389_2020136398_731_20230824-111014/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136416_2020136425_774_20230824-115130/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/RT_Cine_LIN_STCNNT_41837_2020136416_2020136425_774_20230824-115130/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

python3 ./projects/mri/inference/run_inference.py --input_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/LGE_MOCO_AVE_STCNNT_41837_2020136416_2020136425_772_20230824-115018/DebugOutput --output_dir /isilon/lab-kellman/ReconResults/Barts_STCNNT/20230824/LGE_MOCO_AVE_STCNNT_41837_2020136416_2020136425_772_20230824-115018/${RES_DIR} --scaling_factor ${scaling_factor} --im_scaling 1.0 --gmap_scaling 1.0 --input_fname input --gmap_fname gmap --saved_model_path $model --model_type ${model_type_str}

```

## Run the LPSF scripts

cd /isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230/scripts

for f in *.sh; do
  bash "$f" 
done


## run the snr test

python3 ./mri/run_inference_snr_pseudo_replica.py --input_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res/DebugOutput/ --output_dir /isilon/lab-xue/data/mri_raw_data/freemax/20230630_NV_AI/meas_MID00164_FID07570_G25_2CH_CINE_256_R4/res_ai/ --im_scaling 1 --gmap_scaling 1 --saved_model_path /isilon/lab-xue/projects/mri/test/hy_search_contined_gaussian_2nd_stage/hy_search_contined_gaussian_2nd_stage_epoch-74.pth --input_fname input --gmap_fname gmap --scaling_factor 1.0 --added_noise_sd 0.1 --rep 32

cases=(
        res_1st_hrnet_TLGTLG_TLGTLG_more_epochs_NN100 
        res_1st_unet_TTT_TTT_NN100 
        res_1st_hrnet_TTT_TTT_NN100 
        res_1st_hrnet_TLG_TLGTLG_NN100
        res_1st_hrnet_TLG_TLGTLG_only_white_noise
        res_1st_hrnet_TLG_TLGTLG_no_gmap
        res_1st_hrnet_TLG_TLG_NN100
        res_1st_unet_TLG_TLG_NN100
        res_1st_hrnet_C3C3C3_C3C3C3_NN100
        res_1st_unet_C3C3C3_C3C3C3_NN100
        res_1st_hrnet_C3C3C3_C3C3C3_only_white_noise
        res_1st_hrnet_C3C3C3_C3C3C3_no_gmap
        res_1st_unet_C3C3C3_C3C3C3_only_white_noise
        res_1st_unet_C3C3C3_C3C3C3_no_gmap
        res_1st_hrnet_C2C2C2_C2C2C2_NN100
        res_1st_unet_C2C2C2_C2C2C2_NN100
    )

base_dir=/isilon/lab-kellman/Share/data/rtcine

cases=(
        res_1st_hrnet_TLGTLG_TLGTLG_more_epochs_NN100 
        res_1st_hrnet_TTT_TTT_NN100 
        res_1st_unet_TTT_TTT_NN100 
        res_1st_hrnet_TLG_TLGTLG_NN100
        res_1st_hrnet_TLG_TLGTLG_only_white_noise
        res_1st_hrnet_TLG_TLGTLG_no_gmap
        res_1st_hrnet_TLG_TLG_NN100
        res_1st_unet_TLG_TLG_NN100
        res_1st_hrnet_C3C3C3_C3C3C3_NN100
        res_1st_unet_C3C3C3_C3C3C3_NN100
        res_1st_hrnet_C3C3C3_C3C3C3_only_white_noise
        res_1st_hrnet_C3C3C3_C3C3C3_no_gmap
        res_1st_unet_C3C3C3_C3C3C3_only_white_noise
        res_1st_unet_C3C3C3_C3C3C3_no_gmap
        res_1st_hrnet_C2C2C2_C2C2C2_NN100
        res_1st_unet_C2C2C2_C2C2C2_NN100
    )
base_dir=/isilon/lab-xue/projects/mri/results/test/mri-MRI_double_net_20230

cases=(
        res_1st_hrnet_C3C3C3_C3C3C3_NN100 
        res_1st_unet_C3C3C3_C3C3C3_NN100 
        res_1st_hrnet_C3C3C3_C3C3C3_only_white_noise 
        res_1st_hrnet_C3C3C3_C3C3C3_no_gmap
        res_1st_unet_C3C3C3_C3C3C3_only_white_noise
        res_1st_unet_C3C3C3_C3C3C3_no_gmap
        res_1st_hrnet_C2C2C2_C2C2C2_NN100
        res_1st_unet_C2C2C2_C2C2C2_NN100
    )
base_dir=/isilon/lab-xue/publications/SNRAware

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"

    for Script in scripts_${cases[$index]}/*.sh ; do
        bash "$base_dir/$Script" &
    done
done



for Script in *LPSF*.sh ; do
    echo bash "$base_dir/$Script" &
done

for index in ${!cases[*]}; do 
    echo "${cases[$index]}"

    for Script in scripts_${cases[$index]}/*LPSF*.sh ; do
        echo bash "$base_dir/$Script" &
    done

done
