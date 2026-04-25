import os


IMAGE_DIR = ''
DATASET_PATH = ''
RESULTS_PATH = ''
FONT_DIR = ''


IMG_SIZE = (224, 224)
REPROMPT_TIMES = 5

# === Prompts ===
PROMPTS = {
    "recognition": "Identify the {super_category} breed of the animal pictured in the image. Answer with the breed name directly.",
    "rep_attr": "Output a list of the typical attributes of this {super_category} breed, expressed strictly as adjectives.",
    "dis_attr": "Output a list of attributes that distinguish this {super_category} breed from other {super_category} breeds, expressed strictly as adjectives.",
    "adj_rep_attr": "Output a list of adjectives that describe this {super_category} breed.",
    "adj_dis_attr": "Output a list of adjectives that capture how this {super_category} breed is different from other {super_category} breeds.",
    "reword_rep_attr": "Produce a list of the typical characteristics of this {super_category} breed, expressed strictly as adjectives.",
}

# === font ===
FONTS = {
    "sans_serif": ["Arial.ttf", "calibril.ttf", "Consolas.ttf", "Helvetica.ttf", "Vera.ttf", "Futura.ttf", "Verdana.ttf", "GillSans.ttf"],
    "cursive": ["Brush.ttf", "edwardianscriptitc.ttf", "FREESCPT.ttf", "FRSCRIPT.TTF", "Lucida_Handwriting.TTF", "Magneto.TTF", "Mistral.TTF", "Segoe_script.ttf" ]
}

# === breeds ===
BREEDS_INFO = [
    ("American Bulldog","dog"),
    ("American Pit Bull Terrier","dog"),
    ("Basset Hound","dog"),
    ("Beagle","dog"),
    ("Boxer","dog"),
    ("Chihuahua","dog"),
    ("English Setter","dog"),
    ("German Shorthaired","dog"),
    ("Great Pyrenees","dog"),
    ("Havanese","dog"),
    ("Japanese Chin","dog"),
    ("Keeshond","dog"),
    ("Leonberger","dog"),
    ("Miniature Pinscher","dog"),
    ("Pomeranian","dog"),
    ("Pug","dog"),
    ("Saint Bernard","dog"),
    ("Samoyed","dog"),
    ("Scottish Terrier","dog"),
    ("Shiba Inu","dog"),
    ("Staffordshire Bull Terrier","dog"),
    ("Wheaten Terrier","dog"),
    ("Yorkshire Terrier","dog"),
    ("Abyssinian","cat"),
    ("Bengal", "cat"),
    ("Bombay", "cat"),
    ("British Shorthair", "cat"),
    ("Egyptian Mau", "cat"),
    ("Maine Coon", "cat"),
    ("Ragdoll", "cat"),
    ("Russian Blue", "cat"),
    ("Siamese", "cat"),
    # exculded breeds for textimg 
    # ("English Cocker Spaniel","dog"),
    # ("Newfoundland","dog"),
    # ("Birman", "cat"),
    # ("Persian", "cat"),
    # ("Sphynx", "cat")
]

