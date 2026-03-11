import os

instance_dir = "data/instances/WIP-FMS"

mapping = {
"WIP-FMS_small_S3_K2_N20_balanced_seed55.json":"WIP-FMS_01.json",
"WIP-FMS_small_S3_K2_N20_balanced_seed66.json":"WIP-FMS_02.json",
"WIP-FMS_small_S3_K2_N20_downstream_bottleneck_seed11.json":"WIP-FMS_03.json",
"WIP-FMS_small_S3_K2_N20_downstream_bottleneck_seed22.json":"WIP-FMS_04.json",
"WIP-FMS_small_S3_K2_N20_mid_bottleneck_seed33.json":"WIP-FMS_05.json",
"WIP-FMS_small_S3_K2_N20_mid_bottleneck_seed44.json":"WIP-FMS_06.json",

"WIP-FMS_mid_S5_K3_N60_balanced_seed66.json":"WIP-FMS_07.json",
"WIP-FMS_mid_S5_K3_N60_balanced_seed77.json":"WIP-FMS_08.json",
"WIP-FMS_mid_S5_K3_N60_downstream_bottleneck_seed11.json":"WIP-FMS_09.json",
"WIP-FMS_mid_S5_K3_N60_downstream_bottleneck_seed22.json":"WIP-FMS_10.json",
"WIP-FMS_mid_S5_K3_N60_downstream_bottleneck_seed33.json":"WIP-FMS_11.json",
"WIP-FMS_mid_S5_K3_N60_mid_bottleneck_seed44.json":"WIP-FMS_12.json",
"WIP-FMS_mid_S5_K3_N60_mid_bottleneck_seed55.json":"WIP-FMS_13.json",

"WIP-FMS_large_S8_K4_N120_balanced_seed66.json":"WIP-FMS_14.json",
"WIP-FMS_large_S8_K4_N120_balanced_seed77.json":"WIP-FMS_15.json",
"WIP-FMS_large_S8_K4_N120_downstream_bottleneck_seed11.json":"WIP-FMS_16.json",
"WIP-FMS_large_S8_K4_N120_downstream_bottleneck_seed22.json":"WIP-FMS_17.json",
"WIP-FMS_large_S8_K4_N120_downstream_bottleneck_seed33.json":"WIP-FMS_18.json",
"WIP-FMS_large_S8_K4_N120_mid_bottleneck_seed44.json":"WIP-FMS_19.json",
"WIP-FMS_large_S8_K4_N120_mid_bottleneck_seed55.json":"WIP-FMS_20.json",
}

for old_name,new_name in mapping.items():

    old_path = os.path.join(instance_dir,old_name)
    new_path = os.path.join(instance_dir,new_name)

    if os.path.exists(old_path):

        os.rename(old_path,new_path)

        print(old_name," -> ",new_name)

print("rename finished")