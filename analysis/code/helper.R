#' Function to match subject verb target with corresponding id
#' 0-indexed list (make sure to add one in R)
#'
#' @param object choose from {subject, verb, target}
#' @param idx id for the chosen subject, verb, or target 
idx_to_component_name <- function(object, idx){
  
  subjects = c(
    'Endocatch bag',        # 1
    'Endoloop ligature',    # 2
    'Endoscopic stapler',   # 3
    'Grasper',              # 4
    'Gauze',                # 5
    'Hemoclip',             # 6
    'Hemoclip applier',     # 7
    'Hemostatic agents',    # 8
    'Kittner',              # 9 
    'L-hook electrocautery',# 10
    'Maryland dissector',   # 11
    'Needle',               # 12
    'Port',                 # 13
    'Scissors',             # 14
    'Suction irrigation',   # 15
    'Unknown instrument'    # 16
  )
  
  verbs = c(
    'Aspirate',             # 1
    'Avulse',               # 2
    'Clip',                 # 3
    'Coagulate',            # 4
    'Cut',                  # 5
    'Puncture',             # 6
    'Dissect',              # 7
    'Grasp',                # 8
    'Suction/Irrigation',   # 9 
    'Pack',                 # 10
    'Retract',              # 11
    'Null-verb',            # 12
    'Tear'                  # 13
  )
  
  targets = c(
    'Black background',          # 1
    'Abdominal wall',            # 2
    'Adhesion',                  # 3
    'Bile',                      # 4
    'Blood',                     # 5
    'Connective tissue',         # 6
    'Cystic artery',             # 7
    'Cystic duct',               # 8
    'Cystic pedicle',            # 9
    'Cystic plate',              # 10
    'Falciform ligament',        # 11
    'Fat',                       # 12
    'Gallbladder',               # 13
    'Gallstone',                 # 14
    'GI tract',                  # 15
    'Hepatoduodenal ligament',   # 16
    'Liver',                     # 17
    'Omentum',                   # 18
    'Unknown anatomy'            # 19
  )
  
  if (object == "subject"){
    return(subjects[idx])
  } else if (object == "verb"){
    return(verbs[idx])
  } else {
    return(targets[idx])
  }
  
}


#' Function to translate unique id to action triplet name.
#' #' e.g. dur_unique_4.8.12 -> grasper grasp fat
#'
#' @param idx_vector a vector of action triplet idx 
idx_to_name_triplet <- function(idx_vector){
  
  index_cnt_dur <- grep("^cnt_unique|^dur_unique", idx_vector)
  feature_names <- idx_vector[index_cnt_dur]
  
  # map key to name
  mapped_names <- c()
  feature_triplet <- c()
  feature_id <- c()
  
  for (i in 1:length(feature_names)){
    feature <- strsplit(feature_names[i], "_")[[1]]
    feature_triplet <- strsplit(feature[3], "\\.")[[1]]
    feature_subject <- idx_to_component_name("subject", as.numeric(feature_triplet[1])+1)
    feature_verb <- idx_to_component_name("verb", as.numeric(feature_triplet[2])+1)
    feature_target <- idx_to_component_name("target", as.numeric(feature_triplet[3])+1)
    mapped <- paste0(feature_subject, " ", feature_verb, " ", feature_target)
    mapped_names[i] <- mapped
  }
  
  return(mapped_names)
}


#' Function to translate any idx to any variable name.
#' Can include other strings, will remain untouched.
#'
#' @param idx_vector a vector of idx
idx_to_name <- function(idx_vector){
  
  feature_names <- idx_vector
  index_triplet <- grep("^cnt_unique|^dur_unique", idx_vector)
  triplet_names <- idx_vector[index_triplet]
  feature_names[index_triplet] <- idx_to_name_triplet(triplet_names)
  
  return(feature_names)
}