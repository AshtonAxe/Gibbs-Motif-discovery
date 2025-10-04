# Load synthetic data
!wget -c "https://www.dropbox.com/scl/fo/jusdkz3vp19piw33zypnv/AAfFYCmBI96sQBMfmbRLOCg?rlkey=dr0oty4gk1tgr2u3wwme7jkjx&st=zqp0eglz&dl=0" -O gibbs_motifs.zip
!unzip -o gibbs_motifs.zip -d gibbs_motifs
os.unlink("gibbs_motifs.zip")
gibbs_motifs = {}
for fname in os.listdir("gibbs_motifs"):
    with open(os.path.join("gibbs_motifs", fname), "r") as R:
        gibbs_motifs[fname] = R.read().strip().split()
