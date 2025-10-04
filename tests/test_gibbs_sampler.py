def get_motif_seq(PWM):
    motif_seq = ""

    for i in range(len(PWM[0])):
        comp = [row[i] for row in PWM]
        ind = max(range(len(comp)), key = comp.__getitem__)
        motif_seq += alphabet[ind]

    return motif_seq

def print_PWM(PWM, L):
    PWM_comp = []
    PWM_comp.append("||" + "|".join([str(i) for i in range(1, L + 1)]) + "|")
    PWM_comp.append("|-" * (L + 1) + "|")

    for i in range(4):
        PWM_comp.append("|" + alphabet[i] + "|" + "|".join([str(round(val, 2)) for val in PWM[i]]) + "|")

    print("\n".join(PWM_comp) + "\n")

def print_logo(PWM):
    PWM = np.array(PWM).T
    PWM = pd.DataFrame(PWM, columns = alphabet)
    logomaker.Logo(PWM, fade_below = 0.8)

def run_GibbsSampler(S, L, n):
    motif_seqs, motif_PWM = {}, {}

    for i in range(n):
        PWM = GibbsSampler(S, L)
        motif_seq = get_motif_seq(PWM)
        motif_seqs.setdefault(motif_seq, 0)
        motif_seqs[motif_seq] += 1
        motif_PWM[motif_seq] = PWM

    best_motif = max(motif_seqs.keys(), key = lambda x: motif_seqs[x])
    print("Most consistent motif: \n" + best_motif + "\n")
    print("PWM (paste into text block):")
    print_PWM(motif_PWM[best_motif], L)
    print("Sequence logo:")
    print_logo(motif_PWM[best_motif])

# Load synthetic data
!wget -c "https://www.dropbox.com/scl/fo/jusdkz3vp19piw33zypnv/AAfFYCmBI96sQBMfmbRLOCg?rlkey=dr0oty4gk1tgr2u3wwme7jkjx&st=zqp0eglz&dl=0" -O gibbs_motifs.zip
!unzip -o gibbs_motifs.zip -d gibbs_motifs
os.unlink("gibbs_motifs.zip")
gibbs_motifs = {}
for fname in os.listdir("gibbs_motifs"):
    with open(os.path.join("gibbs_motifs", fname), "r") as R:
        gibbs_motifs[fname] = R.read().strip().split()

run_GibbsSampler(gibbs_motifs["data1.txt"], 10, 100)
run_GibbsSampler(gibbs_motifs["data2.txt"], 10, 100)
run_GibbsSampler(gibbs_motifs["data3.txt"], 10, 100)
run_GibbsSampler(gibbs_motifs["data4.txt"], 10, 100)
