import os

os.system("kaggle datasets download -d greatgamedota/lyft-scenes")
os.system("unzip -q lyft-scenes.zip -d 'data/'")
os.system("rm lyft-scenes.zip")

os.system("kaggle datasets download -d greatgamedota/lyft-aerial-and-semantic-maps")
os.system("unzip -q lyft-aerial-and-semantic-maps.zip -d 'data/'")
os.system("rm lyft-aerial-and-semantic-maps.zip")

os.system("kaggle datasets download -d greatgamedota/lyft-metajson")
os.system("unzip -q lyft-metajson.zip -d 'data/'")
os.system("rm lyft-metajson.zip")

os.system("kaggle datasets download -d greatgamedota/lyft-validate-chopped")
os.system("unzip -q lyft-validate-chopped.zip -d 'data/validate_chopped_100/'")
os.system("rm lyft-validate-chopped.zip")