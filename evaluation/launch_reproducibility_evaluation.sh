DATASET_FOLDER="path/to/dataset/"

printf "1.June20 - row3 (INTRA - CENTERS3D)\n"
st_mapping-semantic_mapping $DATASET_FOLDER/1.June20/row3 --config config/default.yaml
printf "\n\n\n 2.June22 using 1.June20 as reference - row3 (INTER - CENTERS3D)\n"
st_mapping-semantic_mapping_onref $DATASET_FOLDER/2.June22/row3 $DATASET_FOLDER/1.June20/row3 --config config/default.yaml
printf "\n\n\n 5.July07 using 1.June20 as reference - row3 (INTER - CENTERS3D)\n"
st_mapping-semantic_mapping_onref $DATASET_FOLDER/5.July07/row3 $DATASET_FOLDER/1.June20/row3 --config config/default.yaml

printf "\n\n\n 1.June20 - row4 (INTRA - CENTERS3D)\n"
st_mapping-semantic_mapping $DATASET_FOLDER/1.June20/row4 --config config/default.yaml
printf "\n\n\n 2.June22 using 1.June20 as reference - row4 (INTER - CENTERS3D)\n"
st_mapping-semantic_mapping_onref $DATASET_FOLDER/2.June22/row4 $DATASET_FOLDER/1.June20/row4 --config config/default.yaml

printf "\n\n\n 2.June22 - row3 (INTRA - CENTERS3D)\n"
st_mapping-semantic_mapping $DATASET_FOLDER/2.June22/row3 --config config/default.yaml
printf "\n\n\n 5.July07 using 2.June22 as reference - row3 (INTER - CENTERS3D)\n"
st_mapping-semantic_mapping_onref $DATASET_FOLDER/5.July07/row3 $DATASET_FOLDER/2.June22/row3 --config config/default.yaml
