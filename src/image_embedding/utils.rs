use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array, Array3};
use std::ops::{Div, Sub};
#[cfg(feature = "hf-hub")]
use std::{fs::read_to_string, path::Path};

// Represents the data being transformed, can be image or ndarray
pub enum TransformData {
    Image(DynamicImage),
    NdArray(Array3<f32>),
}

impl TransformData {
    pub fn image(self) -> anyhow::Result<DynamicImage> {
        match self {
            TransformData::Image(img) => Ok(img),
            _ => Err(anyhow!(
                "TransformData type error: Expected Image, found NdArray"
            )),
        }
    }

    pub fn array(self) -> anyhow::Result<Array3<f32>> {
        match self {
            TransformData::NdArray(array) => Ok(array),
            _ => Err(anyhow!(
                "TransformData type error: Expected NdArray, found Image"
            )),
        }
    }
}

// Trait for any image transformation step
pub trait Transform: Send + Sync {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData>;
}

// --- Individual Transforms ---

struct ConvertToRGB;
impl Transform for ConvertToRGB {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let image = data.image()?;
        // Ensure image is RGB8 for consistency
        let image = image.into_rgb8().into();
        Ok(TransformData::Image(image))
    }
}

// Enum to define the resizing intent based on config
#[derive(Debug, Clone, Copy)]
enum ResizeIntent {
    ShortestEdge(u32),
    Exact { width: u32, height: u32 },
}

// Resize struct now holds the intent and resample filter
pub struct Resize {
    intent: ResizeIntent,
    resample: FilterType,
}

// Helper function to calculate output size maintaining aspect ratio
fn calculate_aspect_ratio_size(
    current_width: u32,
    current_height: u32,
    target_shortest_edge: u32,
) -> (u32, u32) {
    if current_width == 0 || current_height == 0 {
        return (0, 0); // Avoid division by zero
    }

    let scale = if current_width < current_height {
        target_shortest_edge as f64 / current_width as f64
    } else {
        target_shortest_edge as f64 / current_height as f64
    };

    let new_width = (current_width as f64 * scale).round() as u32;
    let new_height = (current_height as f64 * scale).round() as u32;

    (new_width, new_height)
}

impl Transform for Resize {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let image = data.image()?;
        let (current_width, current_height) = image.dimensions();

        // Calculate the final output dimensions based on the intent
        let (output_width, output_height) = match self.intent {
            ResizeIntent::ShortestEdge(edge) => {
                calculate_aspect_ratio_size(current_width, current_height, edge)
            }
            ResizeIntent::Exact { width, height } => (width, height),
        };

        // Perform the actual resize ONLY if dimensions change and are valid
        if output_width > 0
            && output_height > 0
            && (output_width != current_width || output_height != current_height)
        {
            let resized_image = image.resize_exact(output_width, output_height, self.resample);
            Ok(TransformData::Image(resized_image))
        } else {
            Ok(TransformData::Image(image)) // Return original if no resize needed or invalid target
        }
    }
}

pub struct CenterCrop {
    // Store as width, height consistent with image crate
    size: (u32, u32),
}

impl Transform for CenterCrop {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let image = data.image()?; // Expects image data before cropping
        let (origin_width, origin_height) = image.dimensions();
        let (crop_width, crop_height) = self.size;

        // Ensure crop dimensions are valid
        if crop_width == 0 || crop_height == 0 {
            return Err(anyhow!(
                "CenterCrop size cannot be zero: width={}, height={}",
                crop_width,
                crop_height
            ));
        }

        // Check if crop is possible within image bounds
        if crop_width > origin_width || crop_height > origin_height {
            // This case should ideally not happen if resize preceded crop correctly,
            // but handle defensively (or error out)
            // Optionally, you could resize first, or error here depending on desired behavior
            // For now, let's proceed but the result might not be what CLIP expects
            // We might need padding logic here if the spec requires it, but Python likely errors.
        }

        // Calculate top-left corner using integer division (flooring)
        let x = (origin_width.saturating_sub(crop_width)) / 2;
        let y = (origin_height.saturating_sub(crop_height)) / 2;

        // Crop (immutable version)
        let cropped_image = image.crop_imm(
            x,
            y,
            crop_width.min(origin_width),
            crop_height.min(origin_height),
        ); // Use min to avoid panic if warning above occurred
        Ok(TransformData::Image(cropped_image))
        // Note: Removed the complex padding logic from the original snippet,
        // assuming the standard CLIP process resizes *then* crops within bounds.
    }
}

// Converts the DynamicImage to ndarray format (C, H, W)
struct ImageToNdarray;
impl Transform for ImageToNdarray {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        match data {
            TransformData::Image(image) => {
                let image = image.to_rgb8(); // Ensure RGB8
                let (width, height) = image.dimensions();
                let mut pixels_array = Array3::zeros((3usize, height as usize, width as usize));

                for (x, y, pixel) in image.enumerate_pixels() {
                    // Pixel is image::Rgb([u8; 3])
                    pixels_array[[0, y as usize, x as usize]] = pixel[0] as f32; // R
                    pixels_array[[1, y as usize, x as usize]] = pixel[1] as f32; // G
                    pixels_array[[2, y as usize, x as usize]] = pixel[2] as f32;
                    // B
                }
                Ok(TransformData::NdArray(pixels_array))
            }
            TransformData::NdArray(array) => Ok(TransformData::NdArray(array)), // Pass through if already array
        }
    }
}

pub struct Rescale {
    pub scale: f32,
}
impl Transform for Rescale {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let array = data.array()?;
        let array = array * self.scale; // Element-wise multiplication
        Ok(TransformData::NdArray(array))
    }
}

pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}
impl Transform for Normalize {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let array = data.array()?;
        // Ensure mean/std vectors are valid
        if self.mean.len() != 3 || self.std.len() != 3 {
            return Err(anyhow!(
                "Normalize mean and std must have length 3, got mean={}, std={}",
                self.mean.len(),
                self.std.len()
            ));
        }
        // Reshape mean and std for broadcasting (C, 1, 1)
        let mean_arr = Array::from_shape_vec((3, 1, 1), self.mean.clone())?;
        let std_arr = Array::from_shape_vec((3, 1, 1), self.std.clone())?;

        // Check if broadcasting is possible (implicitly handled by ndarray op)
        // let array_normalized = (array - &mean_arr) / &std_arr; // More concise
        let array_normalized = array.sub(&mean_arr).div(&std_arr);

        Ok(TransformData::NdArray(array_normalized))
    }
}

// --- Composition and Loading ---

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    #[cfg(feature = "hf-hub")]
    pub fn from_file<P: AsRef<Path>>(file: P) -> anyhow::Result<Self> {
        let content = read_to_string(file)?;
        let config = serde_json::from_str(&content)?;
        load_preprocessor(config)
    }

    // Load config directly from bytes (e.g., embedded resource)
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> anyhow::Result<Compose> {
        let config = serde_json::from_slice(bytes.as_ref())?;
        load_preprocessor(config)
    }
}

// Apply the sequence of transforms
impl Transform for Compose {
    fn transform(&self, mut data: TransformData) -> anyhow::Result<TransformData> {
        for transform in &self.transforms {
            data = transform.transform(data)?;
        }
        Ok(data)
    }
}

// Function to load the pipeline from JSON config
fn load_preprocessor(config: serde_json::Value) -> anyhow::Result<Compose> {
    let mut transformers: Vec<Box<dyn Transform>> = vec![];

    // 1. Convert to RGB (Always done first for consistency)
    if config["do_convert_rgb"].as_bool().unwrap_or(true) {
        transformers.push(Box::new(ConvertToRGB));
    }

    let mode = config["image_processor_type"]
        .as_str()
        .ok_or_else(|| anyhow!("'image_processor_type' not found or not a string in config"))?;

    match mode {
        "CLIPImageProcessor" => {
            // 2. Resize
            if config["do_resize"].as_bool().unwrap_or(true) {
                // Default true for CLIP
                let size_config = config["size"].clone();
                if size_config.is_null() {
                    return Err(anyhow!("'size' configuration is missing for resize"));
                }

                let resize_intent = if let Some(edge) = size_config["shortest_edge"].as_u64() {
                    ResizeIntent::ShortestEdge(edge as u32)
                } else if let (Some(h), Some(w)) = (
                    size_config["height"].as_u64(),
                    size_config["width"].as_u64(),
                ) {
                    ResizeIntent::Exact {
                        width: w as u32,
                        height: h as u32,
                    }
                } else {
                    return Err(anyhow!(
                        "'size' must contain either 'shortest_edge' or both 'height' and 'width'."
                    ));
                };

                // Read and Map Resample Filter
                let resample_value = config["resample"].as_u64().unwrap_or(3); // Default 3 (Bicubic)
                                                                               // println!("Config requests resample value: {}", resample_value); // Debugging

                let filter_type = match resample_value {
                    0 => FilterType::Nearest,
                    1 => FilterType::Lanczos3,
                    2 => FilterType::Triangle, // Bilinear in Pillow
                    // *** Mapping 3 (Bicubic) to CatmullRom ***
                    // This is the closest cubic filter in `image` crate, but NOT identical to Pillow's Bicubic.
                    3 => FilterType::CatmullRom,
                    4 => FilterType::Gaussian, // Map Box? Gaussian is reasonable default.
                    5 => FilterType::Lanczos3, // Map Hamming? Lanczos3 is reasonable default.
                    _ => {
                        eprintln!(
                            "Warning: Unknown resample value {}, defaulting to CatmullRom",
                            resample_value
                        );
                        FilterType::CatmullRom // Default fallback
                    }
                };
                // println!("Using FilterType: {:?}", filter_type); // Debugging

                transformers.push(Box::new(Resize {
                    intent: resize_intent,
                    resample: filter_type,
                }));
            }

            // 3. Center Crop
            if config["do_center_crop"].as_bool().unwrap_or(true) {
                // Default true for CLIP
                let crop_size_config = config["crop_size"].clone();
                if crop_size_config.is_null() {
                    return Err(anyhow!(
                        "'crop_size' configuration is missing for center_crop"
                    ));
                }

                // Read crop size (can be single int or object with height/width)
                let (crop_height, crop_width) = if let Some(size) = crop_size_config.as_u64() {
                    (size as u32, size as u32) // Square crop if single int
                } else if crop_size_config.is_object() {
                    (
                        crop_size_config["height"]
                            .as_u64()
                            .map(|h| h as u32)
                            .ok_or(anyhow!("'crop_size' object missing 'height'"))?,
                        crop_size_config["width"]
                            .as_u64()
                            .map(|w| w as u32)
                            .ok_or(anyhow!("'crop_size' object missing 'width'"))?,
                    )
                } else {
                    return Err(anyhow!(
                        "Invalid 'crop_size' format: {:?}",
                        crop_size_config
                    ));
                };

                transformers.push(Box::new(CenterCrop {
                    // Pass size as (width, height) consistent with image crate
                    size: (crop_width, crop_height),
                }));
            }
        }
        // Add other processor types here if needed ("ConvNextFeatureExtractor", "BitImageProcessor", etc.)
        // Ensure their resize logic also considers aspect ratio correctly if needed.
        mode => return Err(anyhow!("Unsupported image_processor_type: {}", mode)),
    }

    // 4. Convert Image to Ndarray (C, H, W) format for numerical ops
    transformers.push(Box::new(ImageToNdarray));

    // 5. Rescale pixel values (e.g., 0-255 -> 0-1)
    if config["do_rescale"].as_bool().unwrap_or(true) {
        // Default true for CLIP
        let rescale_factor = config["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0); // Default 1/255
        transformers.push(Box::new(Rescale {
            scale: rescale_factor as f32,
        }));
    }

    // 6. Normalize
    if config["do_normalize"].as_bool().unwrap_or(true) {
        // Default true for CLIP
        // Helper to parse float arrays from JSON
        let parse_float_array = |key: &str| -> Result<Vec<f32>> {
            config[key]
                .as_array()
                .ok_or_else(|| anyhow!("'{}' array is missing or not an array", key))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|n| n as f32)
                        .ok_or_else(|| anyhow!("Non-float value found in '{}' array", key))
                })
                .collect()
        };

        let mean = parse_float_array("image_mean")?;
        let std = parse_float_array("image_std")?;

        transformers.push(Box::new(Normalize { mean, std }));
    }

    Ok(Compose::new(transformers))
}
