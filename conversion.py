# Here's the train set and test set.
# https://pjreddie.com/projects/mnist-in-csv/

# The format is:
# 
#     label, pix-11, pix-12, pix-13, ...
# 
# where pix-ij is the pixel in the ith row and jth column.
# 
# For the curious, this is the script to generate the csv files from the original data.


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte",
        "./data/mnist_train.csv", 60000)
convert("./data/t10k-images-idx3-ubyte", "./data/t10k-labels-idx1-ubyte",
        "./data/mnist_test.csv", 10000)

#Puis depuis un terminal, depuis le dossier data, lancer les commandes :
#    cat mnist_train.csv > mnist_complet.csv
#    cat mnist_test.csv >> mnist_complet.csv



