alphabet = ["A", "C", "G", "T"]
alphabet_dict = {'A' : 0, 'C' : 1, 'G' : 2, 'T' :3 }
B = [0.25, 0.25, 0.25, 0.25]

def get_PWM(S, starts, L):
    """
    Build a Position Weight Matrix from sequences and their motif starting positions.

    This function extracts motif subsequences from each sequence, counts nucleotide
    frequencies at each position, and converts to probabilities with pseudocounts.

    Args:
        S (list): List of DNA sequences as strings
        starts (list): Starting position of motif in each sequence (same length as S)
        L (int): Length of motif

    Returns:
        list: 4 x L matrix where PWM[nucleotide_index][position] = probability
              Nucleotide indices: A=0, C=1, G=2, T=3

    Algorithm:
        1. Initialize count matrix: PWM_counts[nucleotide][position] = 0
        2. For each sequence, extract motif starting at given position
        3. Count each nucleotide at each position in the motif
        4. Convert counts to probabilities with +1 pseudocounts:
           PWM[nucleotide][position] = (count + 1) / (num_sequences + 4)

    Example:
        >>> sequences = ["ATCGAA", "GCTTAA"]
        >>> positions = [2, 0]  # Motifs: "CGAA", "GCTT"
        >>> get_PWM(sequences, positions, 4)
        # Returns PWM for aligned motifs "CGAA" and "GCTT"

    Hint: Use alphabet_dict to convert nucleotides to indices (A->0, C->1, G->2, T->3)
    """

    # Create PWN_counts matrix
    PWM_counts = torch.zeros(4, L, dtype=torch.float)
    for position, sequence in enumerate(S):
      for index, nucleotide in enumerate(sequence[starts[position]: starts[position] + L]):
        nucleotide_index = alphabet_dict[nucleotide]
        PWM_counts[nucleotide_index, index] += 1.0

    # Create PWM
    PWM = (PWM_counts + 1) / (len(S) + 4)

    return PWM.tolist()

def get_new_position(PWM, s, L):
    """
    Sample a new motif starting position in a sequence using likelihood ratios.

    This function scans all possible L-length subsequences of the sequence,
    scores each subsequence based on how well it matches the motif PWM relative
    to background, converts scores into probabilities, and samples one position
    proportional to those probabilities.

    Args:
        PWM (list): 4 x L Position Weight Matrix built from other sequences
        s (str): DNA sequence to scan for motif
        L (int): Length of motif

    Returns:
        int: Starting position of the newly sampled L-length subsequence

    Algorithm:
        1. For each possible starting position i (0 through len(s) - L):
           - Extract the k-mer s[i:i+L]
           - Compute likelihood ratio score relative to background
        2. Normalize all scores into a probability distribution
        3. Randomly sample one position according to these probabilities
        4. Return the sampled position

    Scoring:
        For each nucleotide in the k-mer:
            score *= PWM[nucleotide][position] / B[nucleotide]
        where B[nucleotide] = 0.25 (uniform background probability)

    Example:
        >>> PWM = [[0.5, 0.1], [0.1, 0.1], [0.3, 0.1], [0.1, 0.7]]  # Favors "AT"
        >>> get_new_position(PWM, "CGATCC", 2)
        1  # Likely to return position of "AT", but result may vary due to sampling
    """
    starts = len(s) - L + 1
    scores = []
    for i in range(starts):
      k_mer = s[i: i+L]
      score = 1
      for position, nucleo in enumerate(k_mer):
        nucleo_num = alphabet_dict[nucleo]
        score *= PWM[nucleo_num][position] / 0.25
      scores.append(score)

    # Normalize all scores
    probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()

    # Randomly sample one position from probabilities
    items = range(starts)
    sample = random.choices(items, weights=probabilities, k=1)[0]

    return sample

def GibbsSampler(S, L):
    """
    Implement the Gibbs sampling algorithm for motif finding.

    The Gibbs sampler iteratively refines motif positions by:
    1. Randomly initializing motif starting positions in each sequence
    2. For a fixed number of iterations, randomly selecting one sequence at a time
    3. Removing the selected sequence from consideration and calculating PWM from remaining sequences
    4. Using the PWM to sample a new starting position for the selected sequence
    5. Repeating for a maximum of 50 iterations

    Args:
        S (list): List of DNA sequences as strings
        L (int): Length of motif to find

    Returns:
        list: Final Position Weight Matrix (4 x L) representing the discovered motif
    """
    starting_positions = [random.randrange(len(S[i]) - L + 1) for i in range(len(S))]

    for num in range(50):
      sequence_index = random.choice(range(len(S)))
      list_without_sequence = S[:sequence_index] + S[sequence_index + 1:]
      new_starting_positions = starting_positions[:sequence_index] + starting_positions[sequence_index + 1:]
      PWM = get_PWM(list_without_sequence, new_starting_positions, L)

      new_position = get_new_position(PWM, S[sequence_index], L)

      starting_positions[sequence_index] = new_position

    return get_PWM(S, starting_positions, L)

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
