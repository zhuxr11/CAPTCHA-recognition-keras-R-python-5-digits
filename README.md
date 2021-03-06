
# Breaking Text-Based CAPTCHA with Convolutional Nerual Network (CNN)

**Author**: Xiurui Zhu<br /> **Modified**: 2021-11-23 10:02:42<br />
**Compiled**: 2021-11-23 10:02:45

## Abstract

CAPTCHA is widely used to detect automated spamming on websites. In
recent past, CAPTCHA images usually text-based, consisting of digits and
letters with proper distortion, blurring and noise. With the development
of deep learning, these CAPTCHA images become breakable with
convolutional neural network (CNN), as demonstrated in python. This
paper attempted the process of breaking 5-digit CAPTCHA images in R with
940 samples as training dataset and another 100 ones as testing dataset,
achieving an accuracy of 70%. With the successful prediction of the
CAPTCHA images, more possibilities and challenges were suggested for
further thinking.

## Introduction

CAPTCHA stands for “**C**ompletely **A**utomated **P**ublic **T**uring
test to tell **C**omputers and **H**umans **A**part”. There are mainly
two kinds of CAPTCHA systems, the text-based one and the image-based
one. The text-based CAPTCHA is the earlier version that usually contains
a known number of digits and letters. To escape the detection by optical
character recognition (OCR), the text-based CAPTCHA images usually
contains distortion, blurring and noise (such as random deletion lines).
The text-based images are now being depricated, since they are known to
be breakable by deep learning technology, such as convolutional neural
network (CNN), as demonstrated by [a study in
python](https://medium.com/@manvi./captcha-recognition-using-convolutional-neural-network-d191ef91330e).
This paper will attempt this process in a mixture of R and python.

## Methods

### Data preparation

To facilitate the analyses in the paper, we need to load the following
packages in R: `tidyverse`, `magrittr`, `rlang`, `reticulate`, `png`,
`tools`, `ggpubr` and `ggtext`. We also need the following packages
installed in python: `numpy`, `tensorflow`, `keras` and `pydot`.
Furthermore, we need [graphviz](https://graphviz.gitlab.io/download/) to
visualize model structure.

``` r
# Define a function to check, install (if necessary) and load packages
check_packages <- function(pkg_name, repo = c("cran", "github"), repo_path) {
  repo <- match.arg(repo)
  # Load installed packages
  inst_packages <- installed.packages()
  if (pkg_name %in% inst_packages == FALSE) {
    cat("* Installing: ", pkg_name, ", repo = ", repo, "\n", sep = "")
    switch(repo,
           cran = install.packages(pkg_name),
           github = {
             if ("devtools" %in% inst_packages == FALSE) {
               install.packages("devtools")
             }
             devtools::install_github(repo_path)
           })
  } else {
    cat("* Package already installed: ", pkg_name, "\n", sep = "")
  }
  suppressPackageStartupMessages(
    library(pkg_name, character.only = TRUE)
  )
}

# CRAN packages
check_packages("tidyverse", repo = "cran")
purrr::walk(.x = c("magrittr", "rlang", "reticulate", "png", "tools",
                   "ggpubr", "ggtext"),
            .f = check_packages, repo = "cran")

# Initialize python connection with reticulate
check_python <- function() {
  stopifnot(reticulate::py_available(initialize = TRUE) == TRUE)
}
check_python()
#> * Package already installed: tidyverse
#> * Package already installed: magrittr
#> * Package already installed: rlang
#> * Package already installed: reticulate
#> * Package already installed: png
#> * Package already installed: tools
#> * Package already installed: ggpubr
#> * Package already installed: ggtext
```

``` python
import numpy as np
import pandas as pd
import os
import tensorflow as tf
# Set random seed right after importing tensorflow
tf.random.set_seed(599)
from tensorflow import keras
from tensorflow.keras import layers
import session_info
```

To visualize model structure, we need to handle this process in an
independent python script, since `reticulate` does not facilitate
python-generated plots that need `graphviz`.

``` r
# Define a function that plots model structure with python script
#' @param py_model Python model object, usually as py$<model_name>.
#' @param file_name Output file name for model structure plot (PNG format).
#' @param show_shapes Logical indicating whether layer shapes are shown.
#' @param show_layer_names Logical indicating whether layer names are shown.
#' @param verbose Logical indicating whether detailed messages are printed.
#' @inheritDotParams knitr::include_graphics -path
visualize_model_rmd <- function(
  py_model,
  file_name,
  show_shapes = TRUE,
  show_layer_names = TRUE,
  verbose = TRUE,
  ...
) {
  # Process directory
  if (dir.exists(dirname(file_name)) == FALSE) {
    if (verbose == TRUE) {
      message("* Creating directory: ", dirname(file_name))
    }
    dir.create(dirname(file_name), recursive = TRUE)
  }
  # Get python model variable name (format: c("$", "py", py_model_name))
  py_model_name <- as.character(substitute(py_model)) %>%
    dplyr::last()
  # Save model with python command
  py_model_file_name <- tempfile(fileext = "")
  if (verbose == TRUE) {
    message("* Saving model to: ", py_model_file_name)
  }
  reticulate::py_eval(
    paste0(
      py_model_name,
      ".save(\"",
      py_model_file_name %>%
        stringr::str_replace_all("\\\\", "/"),
      "\")"
    )
  )
  # Write a python script for model visualization
  py_file_name <- tempfile(fileext = ".py")
  if (verbose == TRUE) {
    message("* Writing python script to: ", py_file_name)
  }
  py_command <- paste(
    "import numpy as np",
    "import os",
    "from tensorflow import keras",
    paste0("os.chdir('", getwd(), "')"),
    paste0("model = keras.models.load_model(r'", py_model_file_name, "')"),
    "keras.utils.plot_model(",
    "  model = model,",
    paste0("  to_file = '", file_name, "',"),
    paste0("  show_shapes = ", stringr::str_to_sentence(show_shapes), ","),
    paste0("  show_layer_names = ", stringr::str_to_sentence(show_layer_names)),
    ")",
    "",
    sep = "\n"
  )
  py_file <- file(py_file_name, open = "w")
  write(py_command, py_file)
  close(py_file)
  # Execute the python file
  if (verbose == TRUE) {
    message("* Executing python script...")
  }
  invisible(system(paste0("python ", py_file_name)))
  # Clean up
  if (unlink(py_model_file_name, recursive = TRUE) == 0 && verbose == TRUE) {
    message("* Cleaned up model file: ", py_model_file_name)
  }
  if (unlink(py_file_name) == 0 && verbose == TRUE) {
    message("* Cleaned up script file: ", py_file_name)
  }
  # Include the graphics
  knitr::include_graphics(file_name, ...)
}
```

Image data from a [5-digit text-based CAPTCHA
dataset](https://www.kaggle.com/fournierp/captcha-version-2-images) were
first loaded with the `samples` folder unzipped and placed under the
current working directory. A total of 1040 png images were turned into
grayscale and put into a three-dimensional array where the first one as
samples, the second one as pixel rows and the third as pixel columns.

``` r
# Load image file names
file_names <- list.files("samples",
                         pattern = "\\.png$",
                         full.names = TRUE,
                         recursive = FALSE)

# Load images (this may take minutes)
data_x <- file_names %>%
  purrr::map(~ .x %>%
               png::readPNG() %>%
               # Select the first 3 color channels as RGB
               `[`(, , 1L:3L, drop = FALSE) %>%
               # Turn the image into grayscale
               apply(MARGIN = 1L:2L, mean, na.rm = TRUE) %>%
               reticulate::array_reshape(dim = c(dim(.), 1L))) %>%
  # Turn list into array
  purrr::reduce2(.y = 1L:length(.), 
                 .f = function(array., matrix., idx) {
                   array.[idx, , , ] <- matrix.
                   array.
                 },
                 .init = array(0, dim = c(length(.), dim(.[[1L]]))))
print(dim(data_x))
#> [1] 1040   50  200    1
```

Some sample CAPTCHA image were visualized as below.

``` r
# Define a function to convert matrix to ggplot image
matrix2gg_image <- function(
  matrix.,
  decimal = TRUE,
  title = NULL,
  title_style = ggplot2::element_text(hjust = 0.5),
  plot_margin = grid::unit(c(5.5, 5.5, 5.5, 5.5), "points")
  ) {
  mat_rgb <- matrix. %>%
    apply(MARGIN = 1L:2L, function(x) {
      if (length(x) == 1L) {
        color_chr <- rep(x, 3L)
      } else if (length(x) == 3L) {
        color_chr <- x
      } else {
        stop("The third dimension of matrix. should be 1L or 3L")
      }
      color_chr <- color_chr %>%
        .int2hex_color(decimal = decimal) %>%
        paste(collapse = "") %>%
        {paste0("#", .)}
    })
  plot_data <- mat_rgb %>%
    as.data.frame() %>%
    tibble::rowid_to_column("y") %>%
    tidyr::pivot_longer(cols = !c("y"),
                        names_to = "x",
                        values_to = "fill") %>%
    dplyr::mutate_at("x", ~ .x %>%
                       stringr::str_extract_all("[0-9]+") %>%
                       as.numeric()) %>%
    # Reverse y so that image starts from upper left corner
    dplyr::mutate_at("y", ~ min(.x) + max(.x) - .x)
  plot_obj <- ggplot2::ggplot(plot_data, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_tile(ggplot2::aes(fill = fill),
                       show.legend = FALSE) +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::scale_y_continuous(expand = c(0, 0)) +
    ggplot2::coord_equal(ratio = 1) +
    ggplot2::scale_fill_manual(values = plot_data[["fill"]] %>%
                                 unique() %>%
                                 purrr::set_names(.)) +
    ggplot2::theme_void() +
    ggplot2::theme(plot.margin = plot_margin)
  if (is.null(title) == FALSE) {
    plot_obj +
      ggplot2::ggtitle(title) +
      ggplot2::theme(plot.title = title_style)
  } else {
    plot_obj
  }
}
.int2hex_color <- function(x, decimal = TRUE) {
  if (decimal == TRUE) x <- as.integer(x * 255L)
  stopifnot(is.integer(x) == TRUE)
  x %>%
    as.hexmode() %>%
    as.character() %>%
    stringr::str_pad(width = 2L, pad = "0")
}

# Plot sample images
purrr::reduce(.x = c(5L, 246L, 987L),
              .f = ~ {
                .x[[.y]] <- data_x[.y, , , , drop = TRUE]
                .x
              },
              .init = list()) %>%
  purrr::compact() %>%
  purrr::map(matrix2gg_image, decimal = TRUE, title = NULL) %>%
  {gridExtra::arrangeGrob(grobs = ., nrow = 1L)} %>%
  grid::grid.draw()
```

<img src="README_files/plot-image-1.png" width="100%" />

The labels were then loaded from the file names and turned them into a
list of categorical matrices with one digit per element.

``` r
# Define the number of digits and letters per CAPTCHA
digit <- 5L
# Define a dictionary of digits and letters present in CAPTCHA
class_level <- c(0L:9L, letters)

# Define a helper function for one-hot encoding
to_categorical <- function(idx, class_level) {
  stopifnot(all(idx <= length(class_level)))
  idx %>%
    purrr::map_dfr(~ {
      one_hot <- rep(0L, length(class_level))
      one_hot[.x] <- 1L
      one_hot %>%
        set_names(class_level) %>%
        as.list() %>%
        tibble::as_tibble()
    }) %>%
    `rownames<-`(names(idx)) %>%
    as.matrix()
}
# Define a function to convert character vector to categorical matrix list
labels2matrices <- function(labels, class_level) {
  labels %>%
    stringr::str_extract_all(pattern = ".", simplify = TRUE) %>%
    as.data.frame() %>%
    as.list() %>%
    purrr::set_names(NULL) %>%
    purrr::map(~ {
      factor(.x, levels = class_level) %>%
        as.numeric() %>%
        to_categorical(class_level)
    })
}

# Process image labels
data_y_labels <- file_names %>%
  basename() %>%
  tools::file_path_sans_ext()
data_y <- data_y_labels %>%
  labels2matrices(class_level = class_level)
print(length(data_y))
#> [1] 5
print(dim(data_y[[1L]]))
#> [1] 1040   36
```

### Modeling

A CNN model was built to break the text-based CAPTCHA. A CNN model
consists of two parts, one as convolutional model and the other as deep
neural-network (DNN) model, joined by a flatten layer. Since there are
multiple digits to predict for each CAPTCHA image, we would build the
model including a common convolutional model, a common flatten layer and
multiple DNN models (one for each digit).

``` r
# Define a helper function to transfer variables from r to python
#' @param ... Names of objects to ship to python
r2python <- function(...) {
  check_python()
  var_names <- rlang::enexprs(...) %>%
    as.character()
  purrr::walk2(
    .x = list(...),
    .y = var_names,
    .f = ~ {
      py[[.y]] <- .x
    }
  )
}

# Define a helper function to transfer variables from python to r
#' @param py Python connection created by \code{\link[reticulate]{py_config}}.
#' @param nm Character vector as the names of python variables to ship.
#' @param env Environment to release extracted named \code{py} elements into.
python2r <- function(py, nm, env = rlang::caller_env()) {
  check_python()
  nm %>%
    purrr::walk(~ {
      rlang::env_poke(env, nm = .x, value = py[[.x]])
    })
}
```

``` r
# Wrap up variables and transfer them to python
r2python(digit, class_level, data_x)
```

#### Convolutional model

The convolutional model (diagram as below) was built by adding multiple
modules of convolutional and max-pooling layers, optionally adding a
batch-normalization layer to improve model convergence.

``` python
# Define the convolutional model
input_layer = keras.Input(shape = np.shape(data_x)[1:])
conv_layer = layers.Conv2D(
  filters = 16,
  kernel_size = (3, 3),
  padding = "same",
  activation = "relu"
)(input_layer)
conv_layer = layers.MaxPooling2D(
  pool_size = (2, 2),
  padding = "same"
)(conv_layer)
conv_layer = layers.Conv2D(
  filters = 32,
  kernel_size = (3, 3),
  padding = "same",
  activation = "relu"
)(conv_layer)
conv_layer = layers.MaxPooling2D(
  pool_size = (2, 2),
  padding = "same"
)(conv_layer)
conv_layer = layers.Conv2D(
  filters = 32,
  kernel_size = (3, 3),
  padding = "same",
  activation = "relu"
)(conv_layer)
conv_layer = layers.BatchNormalization()(conv_layer)
conv_layer = layers.MaxPooling2D(
  pool_size = (2, 2),
  padding = "same"
)(conv_layer)
conv_model = keras.Model(inputs = input_layer, outputs = conv_layer)
# Define a flatten layer
conv_layer_flatten = layers.Flatten()(conv_layer)
```

``` r
visualize_model_rmd(
  py_model = py$conv_model,
  file_name = "model_plot/conv_model.png",
  show_shapes = TRUE,
  show_layer_names = FALSE,
  verbose = FALSE
)
```

<img src="model_plot/conv_model.png" width="25%" style="display: block; margin: auto;" />

#### Deep neural network (DNN) models

Each DNN model (diagram as below) was built with a hidden layer and a
dropout layer, with the latter as a regularization method to prevent
overfitting. The output layer of each DNN model adopted a multi-class
configuration with the unit as the number of possibilities per digit and
activation function as `"softmax"`. The input layer of each DNN model
was copied from the shape of the output from the flatten layer.

``` python
# Define a function that copies the shape of a layer and defines an input layer
def build_deep_layer(input_layer, class_level):
  deep_layer = layers.Dense(units = 64, activation = "relu")(input_layer)
  deep_layer = layers.Dropout(rate = 0.5)(deep_layer)
  deep_layer = layers.Dense(
    units = len(class_level),
    activation = "softmax"
  )(deep_layer)
  return deep_layer

# Construct deep model layers (one for each digit)
deep_layers = [
  build_deep_layer(conv_layer_flatten, class_level) for _ in range(digit)
]
```

#### Assembled CNN model

The convolutional model and the DNN models were assembled into a final
CNN model (diagram as below) and the final CNN model was compiled for
training.

``` python
# Construct the final model
model = keras.Model(inputs = input_layer, outputs = deep_layers)
model.summary()
#> Model: "model_1"
#> __________________________________________________________________________________________________
#> Layer (type)                    Output Shape         Param #     Connected to                     
#> ==================================================================================================
#> input_1 (InputLayer)            [(None, 50, 200, 1)] 0                                            
#> __________________________________________________________________________________________________
#> conv2d (Conv2D)                 (None, 50, 200, 16)  160         input_1[0][0]                    
#> __________________________________________________________________________________________________
#> max_pooling2d (MaxPooling2D)    (None, 25, 100, 16)  0           conv2d[0][0]                     
#> __________________________________________________________________________________________________
#> conv2d_1 (Conv2D)               (None, 25, 100, 32)  4640        max_pooling2d[0][0]              
#> __________________________________________________________________________________________________
#> max_pooling2d_1 (MaxPooling2D)  (None, 13, 50, 32)   0           conv2d_1[0][0]                   
#> __________________________________________________________________________________________________
#> conv2d_2 (Conv2D)               (None, 13, 50, 32)   9248        max_pooling2d_1[0][0]            
#> __________________________________________________________________________________________________
#> batch_normalization (BatchNorma (None, 13, 50, 32)   128         conv2d_2[0][0]                   
#> __________________________________________________________________________________________________
#> max_pooling2d_2 (MaxPooling2D)  (None, 7, 25, 32)    0           batch_normalization[0][0]        
#> __________________________________________________________________________________________________
#> flatten (Flatten)               (None, 5600)         0           max_pooling2d_2[0][0]            
#> __________________________________________________________________________________________________
#> dense (Dense)                   (None, 64)           358464      flatten[0][0]                    
#> __________________________________________________________________________________________________
#> dense_2 (Dense)                 (None, 64)           358464      flatten[0][0]                    
#> __________________________________________________________________________________________________
#> dense_4 (Dense)                 (None, 64)           358464      flatten[0][0]                    
#> __________________________________________________________________________________________________
#> dense_6 (Dense)                 (None, 64)           358464      flatten[0][0]                    
#> __________________________________________________________________________________________________
#> dense_8 (Dense)                 (None, 64)           358464      flatten[0][0]                    
#> __________________________________________________________________________________________________
#> dropout (Dropout)               (None, 64)           0           dense[0][0]                      
#> __________________________________________________________________________________________________
#> dropout_1 (Dropout)             (None, 64)           0           dense_2[0][0]                    
#> __________________________________________________________________________________________________
#> dropout_2 (Dropout)             (None, 64)           0           dense_4[0][0]                    
#> __________________________________________________________________________________________________
#> dropout_3 (Dropout)             (None, 64)           0           dense_6[0][0]                    
#> __________________________________________________________________________________________________
#> dropout_4 (Dropout)             (None, 64)           0           dense_8[0][0]                    
#> __________________________________________________________________________________________________
#> dense_1 (Dense)                 (None, 36)           2340        dropout[0][0]                    
#> __________________________________________________________________________________________________
#> dense_3 (Dense)                 (None, 36)           2340        dropout_1[0][0]                  
#> __________________________________________________________________________________________________
#> dense_5 (Dense)                 (None, 36)           2340        dropout_2[0][0]                  
#> __________________________________________________________________________________________________
#> dense_7 (Dense)                 (None, 36)           2340        dropout_3[0][0]                  
#> __________________________________________________________________________________________________
#> dense_9 (Dense)                 (None, 36)           2340        dropout_4[0][0]                  
#> ==================================================================================================
#> Total params: 1,818,196
#> Trainable params: 1,818,132
#> Non-trainable params: 64
#> __________________________________________________________________________________________________
```

``` python
# Compile the final model
model.compile(
  loss = "categorical_crossentropy",
  optimizer = "adam",
  metrics = ["accuracy"]
)
```

``` r
visualize_model_rmd(
  py_model = py$model,
  file_name = "model_plot/final_model.png",
  show_shapes = TRUE,
  show_layer_names = FALSE,
  verbose = FALSE
)
```

<img src="model_plot/final_model.png" width="100%" />

## Results

### Model training

The final CNN model was trained with 940 images with 20% of them as
cross-validation dataset. Please note that in python indices start at 0.

``` r
# Define training and testing dataset
set.seed(999L)
train_idx <- sample.int(dim(data_x)[1L], size = length(file_names) - 100L)
print(length(train_idx))
#> [1] 940
test_idx <- setdiff(seq_along(data_y_labels), train_idx)
print(length(test_idx))
#> [1] 100
# Adjust indices to 0-based
train_idx_0 <- train_idx - 1L
test_idx_0 <- test_idx - 1L

# Subset responses
data_y_train <- data_y %>%
  purrr::map(~ .x[train_idx, , drop = FALSE])
data_y_test <- data_y %>%
  purrr::map(~ .x[test_idx, , drop = FALSE])
```

``` r
# Wrap up variables and transfer them to python
r2python(train_idx_0, test_idx_0, data_y_train, data_y_test)
```

``` python
model_history = model.fit(
  data_x[train_idx_0],
  data_y_train,
  batch_size = 32,
  epochs = 200,
  validation_split = 0.2
)
model_history_df = pd.DataFrame(model_history.history)
```

### Convolutional features

When an image (shown above) went through the convolutional model,
various features were abstracted. For visualization of feature patterns,
the convoluted values were linearly scaled to range \[0,1\] with
positive coefficient and rendered in grayscale (figures as below).

``` python
conv_features = conv_model.predict(x = data_x)
print(np.shape(conv_features))
#> (1040, 7, 25, 32)
```

``` r
python2r(py, "conv_features")
```

``` r
# Select an image
image_idx <- 5L

# Scale selected convolutional features
sel_conv_features_rescale <- conv_features[image_idx, , , , drop = TRUE] %>%
  scales::rescale(to = c(0, 1))
print(dim(sel_conv_features_rescale))
#> [1]  7 25 32

# Convert selected convolutional matrices into images
conv_plots <- purrr::reduce(
  .x = 1:dim(sel_conv_features_rescale)[3L],
  .f = ~ {
    .x[[.y]] <- sel_conv_features_rescale[, , .y, drop = FALSE]
    .x
  },
  .init = list()
) %>%
  purrr::map2(paste0("Feature ", 1:length(.)), ~ {
    .x %>%
      matrix2gg_image(
        decimal = TRUE,
        title = .y,
        title_style = ggplot2::element_text(
          hjust = 0.5,
          size = 10,
          margin = ggplot2::margin(0, 0, 2, 0, unit = "pt")
        ),
        plot_margin = grid::unit(c(0.5, 3.5, 0.5, 3.5), "points")
      )
  })

# Define layout matrix
layout_matrix <- rbind(
  cbind(
    # Original image
    matrix(rep(1L, 4L), nrow = 2L, ncol = 2L),
    # Convolutional features 1~8
    matrix(2L:9L, nrow = 2L, ncol = 4L, byrow = TRUE)
  ),
  # Convolutional features 9~32
  matrix(10L:33L, nrow = 4L, ncol = 6L, byrow = TRUE)
)
print(dim(layout_matrix))
#> [1] 6 6

# Arrange images
data_x[image_idx, , , , drop = TRUE] %>%
  drop() %>%
  reticulate::array_reshape(dim = c(dim(.), 1L)) %>%
  matrix2gg_image(
    decimal = TRUE,
    title = "Original image",
    title_style = ggplot2::element_text(
      hjust = 0.5,
      margin = ggplot2::margin(0, 0, 3, 0, unit = "pt")
    ),
    plot_margin = grid::unit(c(3.5, 3.5, 3.5, 3.5), "points")
  ) %>%
  list() %>%
  append(conv_plots) %>%
  {gridExtra::arrangeGrob(
    grobs = .,
    layout_matrix = layout_matrix,
    heights = grid::unit(rep(3, nrow(layout_matrix)), "line")
  )} %>%
  ggpubr::as_ggplot()
```

<img src="README_files/plot-conv-features-1.png" width="100%" />

### Model performance

Training history of the final CNN model was revealed in terms of loss
and accuracy (figure as below).

``` r
python2r(py, "model_history_df")
```

``` r
# Plot training history: loss and metrics
model_history_df %>%
  tibble::as_tibble() %>%
  dplyr::select(dplyr::matches("dense_")) %>%
  tibble::rowid_to_column("epoch") %>%
  tidyr::pivot_longer(cols = !c("epoch"),
                      names_to = c("model_name", "metric"),
                      names_sep = "(?<=[0-9])_",
                      values_to = "value") %>%
  dplyr::mutate(
    metric_category = ifelse(stringr::str_starts(model_name, "val_"),
                             "Validation",
                             "Training")
  ) %>%
  dplyr::mutate_at(
    "model_name",
    ~ .x %>%
      stringr::str_replace("val_", "") %>%
      stringr::str_replace("dense_", "Model ") %>%
      {
        # Create map from "dense_1/3/5/..." to "Model 1/2/3/..."
        model_idx <- stringr::str_extract(., "[0-9]+") %>%
          as.integer() %>%
          {(. + 1L) / 2L}
        paste0("Model ", model_idx)
      }
  ) %>%
  dplyr::mutate_at("metric", ~ factor(.x, levels = unique(.x))) %>%
  split(f = .[["metric"]]) %>%
  purrr::imap(function(plot_data, metric_name) {
    plot_data %>%
      ggplot2::ggplot(ggplot2::aes(x = epoch, y = value)) +
      ggplot2::geom_line(ggplot2::aes(color = metric_category)) +
      ggplot2::facet_wrap(facets = ggplot2::vars(model_name),
                          nrow = 1L) +
      ggplot2::theme_bw() +
      ggplot2::labs(x = "Epoch",
                    y = stringr::str_to_sentence(metric_name),
                    color = "Category")
  }) %>%
  {ggpubr::ggarrange(plotlist = .,
                     ncol = 1L,
                     align = "hv",
                     labels = "AUTO",
                     legend = "right",
                     common.legend = TRUE)}
```

<img src="README_files/eval-model-perf-1.png" width="100%" />

### Model testing

Tested with the remaining 100 images, the final CNN model achieved an
overall accuracy of 70%.

``` python
model_pred = model.predict(x = data_x[test_idx_0])
```

``` r
python2r(py, "model_pred")
```

``` r
# Define a function to convert categorical matrix list to character vector
matrices2labels <- function(matrices, class_level) {
  matrices %>%
    purrr::map(~ {
      .x %>%
        apply(MARGIN = 1L, function(x) class_level[which.max(x)]) %>%
        as.character()
    }) %>%
    purrr::pmap_chr(paste0)
}

# Derive predictions and convert them to labels
model_pred_labels <- model_pred %>%
  matrices2labels(class_level = class_level)

# Derive overall accuracy
model_accuracy <- purrr::map2_lgl(
  .x = model_pred_labels,
  .y = data_y_labels[test_idx],
  .f = identical
) %>%
  mean()
print(model_accuracy)
#> [1] 0.7
```

Below were the prediction results of some example images from the
testing dataset.

``` r
# Define a function to plot images and print the truth and the prediction
display_pred_example <- function(data, pred, truth, index) {
  # Decide whether the prediction is correct
  pred_correct <- identical(pred[index], truth[index])
  # Format an HTML-style plot title
  plot_title <- paste0(
    "truth: ", truth[index], "<br>",
    "pred : ", "<span style = 'color:",
    if (pred_correct == TRUE) "MediumSeaGreen" else "Tomato", "'>",
    pred[index], "</span>"
  )
  data[index, , , , drop = TRUE] %>%
    matrix2gg_image(
      decimal = TRUE,
      title = plot_title,
      title_style = ggtext::element_markdown(
        family = "mono",
        hjust = 0.5,
        size = 10,
        margin = ggplot2::margin(0, 0, 3, 0, unit = "pt")
      ),
      plot_margin = grid::unit(c(3.5, 3.5, 3.5, 3.5), "points")
    )
}

# Display some prediction results
model_truth_labels <- data_y_labels[test_idx]
model_correct_lgl <- purrr::map2(
  .x = model_pred_labels,
  .y = model_truth_labels,
  .f = identical
)
purrr::map(seq(2L, 97L, by = 5L), ~ {
  display_pred_example(data = data_x[test_idx, , , , drop = FALSE],
                       pred = model_pred_labels,
                       truth = model_truth_labels,
                       index = .x)
}) %>%
  {gridExtra::arrangeGrob(grobs = ., ncol = 5L)} %>%
  ggpubr::as_ggplot()
```

<img src="README_files/test-model-examples-1.png" width="100%" />

## Discussion

In this paper, we presented a CNN in R that predicts text-based CAPTCHA
images at 70% accuracy. The final model was assembled from a common
convolutional module and 5 DNN modules (one for each digit). This
structure is capable of revealing how the final model was trained as a
set of multi-class models, deriving separate loss and accuracy plots for
each digit.

Over the success of predicting 5-digit text-based CAPTCHA, there are
still some food for thought. For example, will the performance of the
final model improve if we unify the DNN models to enable crosstalks
among weight vectors for different digits? Technically, one can use the
following model as a unified DNN model and reshape `data_y` from a list
to an array. At first thought, more information (resulting in more
trainable parameters when printed) is sure to bring up improvements, but
is it really the case (in terms of validation and testing dataset)? And
why?

``` r
# Reshape the responses to an array for the output of unified model
data_y_union <- purrr::reduce(
  .x = 1:length(data_y),
  .f = ~ {
    .x[, .y, ] <- data_y[[.y]]
    .x
  },
  .init = array(dim = dim(data_y[[1L]]) %>%
                  purrr::prepend(length(data_y), 2L))
)

# Send data_y_union to python
r2python(data_y_union)
```

``` python
# Define a unified DNN layer
deep_layer_union = layers.Dense(
  units = 64 * digit,
  activation = "relu"
)(conv_layer_flatten)
deep_layer_union = layers.Dropout(rate = 0.5)(deep_layer_union)
deep_layer_union = layers.Dense(
  units = len(class_level) * digit,
  activation = "linear"
)(deep_layer_union)
deep_layer_union = layers.Reshape(
  target_shape = np.shape(data_y_union)[1:]
)(deep_layer_union)
deep_layer_union = layers.Softmax()(deep_layer_union)

# Define a unified DNN model
model_union = keras.Model(inputs = input_layer, outputs = deep_layer_union)
```

``` r
visualize_model_rmd(
  py_model = py$model_union,
  file_name = "model_plot/final_model_union.png",
  show_shapes = TRUE,
  show_layer_names = FALSE,
  verbose = FALSE
)
```

<img src="model_plot/final_model_union.png" width="25%" style="display: block; margin: auto;" />

Another more challenging exploration is to break text-based CAPTCHA
images without knowing the accurate number of digits. To limit the
complexity of this problem, can we attempt at solving text-based images
with a mixture of 1\~5 digits and/or small letters? Then, how can we
first decide the number of digits in the CAPTCHA image?

## Conclusion

In this paper, a CNN model was built in R to break 5-digit text-based
CAPTCHA. The CNN model comprises a common convolutional model and 5
separate DNN models (one for each digit). The accuracy of the CNN model
on a testing dataset of 100 images was 70% with 200 epochs of training.
Starting from the point of successfully predicting these 5-digit
text-based CAPTCHA images, more structures of the CNN model are worth
exploring and more challenging problems are waiting ahead.

## Session info

This file was compiled with the following packages and versions:

``` r
utils::sessionInfo()
#> R version 4.0.5 (2021-03-31)
#> Platform: x86_64-w64-mingw32/x64 (64-bit)
#> Running under: Windows 10 x64 (build 19042)
#> 
#> Matrix products: default
#> 
#> locale:
#> [1] LC_COLLATE=Chinese (Simplified)_China.936 
#> [2] LC_CTYPE=Chinese (Simplified)_China.936   
#> [3] LC_MONETARY=Chinese (Simplified)_China.936
#> [4] LC_NUMERIC=C                              
#> [5] LC_TIME=Chinese (Simplified)_China.936    
#> 
#> attached base packages:
#> [1] tools     stats     graphics  grDevices utils     datasets  methods  
#> [8] base     
#> 
#> other attached packages:
#>  [1] ggtext_0.1.1    ggpubr_0.4.0    png_0.1-7       reticulate_1.20
#>  [5] rlang_0.4.11    magrittr_2.0.1  forcats_0.5.1   stringr_1.4.0  
#>  [9] dplyr_1.0.7     purrr_0.3.4     readr_2.0.1     tidyr_1.1.3    
#> [13] tibble_3.1.3    ggplot2_3.3.5   tidyverse_1.3.1
#> 
#> loaded via a namespace (and not attached):
#>  [1] httr_1.4.2        jsonlite_1.7.2    carData_3.0-4     modelr_0.1.8     
#>  [5] assertthat_0.2.1  cellranger_1.1.0  yaml_2.2.1        pillar_1.6.2     
#>  [9] backports_1.1.8   lattice_0.20-41   glue_1.4.2        digest_0.6.25    
#> [13] ggsignif_0.6.2    gridtext_0.1.4    rvest_1.0.1       colorspace_1.4-1 
#> [17] cowplot_1.1.1     htmltools_0.5.0   Matrix_1.3-2      pkgconfig_2.0.3  
#> [21] broom_0.7.9       haven_2.4.3       scales_1.1.1      openxlsx_4.2.4   
#> [25] rio_0.5.27        tzdb_0.1.2        farver_2.0.3      generics_0.1.0   
#> [29] car_3.0-11        ellipsis_0.3.2    withr_2.4.1       cli_3.0.1        
#> [33] crayon_1.4.1      readxl_1.3.1      evaluate_0.14     fs_1.5.0         
#> [37] fansi_0.4.2       rstatix_0.7.0     xml2_1.3.2        foreign_0.8-81   
#> [41] data.table_1.13.0 hms_1.1.0         lifecycle_1.0.0   munsell_0.5.0    
#> [45] reprex_2.0.1      zip_2.1.1         compiler_4.0.5    grid_4.0.5       
#> [49] rstudioapi_0.13   rappdirs_0.3.3    labeling_0.3      rmarkdown_2.3    
#> [53] gtable_0.3.0      abind_1.4-5       DBI_1.1.0         curl_4.3         
#> [57] markdown_1.1      R6_2.4.1          gridExtra_2.3     lubridate_1.7.10 
#> [61] knitr_1.29        utf8_1.1.4        stringi_1.4.6     Rcpp_1.0.7       
#> [65] vctrs_0.3.8       dbplyr_2.1.1      tidyselect_1.1.0  xfun_0.15
```

``` python
session_info.show()
#> -----
#> numpy               1.19.5
#> pandas              1.3.4
#> session_info        1.0.0
#> tensorflow          2.5.0
#> -----
#> Python 3.7.11 (default, Jul 27 2021, 09:42:29) [MSC v.1916 64 bit (AMD64)]
#> Windows-10-10.0.19041-SP0
#> -----
#> Session information updated at 2021-11-23 10:17
```
