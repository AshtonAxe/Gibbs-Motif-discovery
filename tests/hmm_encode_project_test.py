!wget -q -c https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz -O chr21.fa.gz
!gunzip -f chr21.fa.gz
!samtools faidx chr21.fa
!wget -q -c https://www.encodeproject.org/files/ENCFF029THO/@@download/ENCFF029THO.bed.gz -O ENCFF029THO.bed.gz

import pyranges as pr
def load_narrow_peak(bed, chr:str=None):
    """
    ENCODE narrowPeak format loader
    """
    df = pd.read_csv(bed, header=None, sep="\t")
    df.columns = ["Chromosome", "Start", "End", "name", "score", "Strand", "signalValue", "pValue", "qValue", "peak"]

    if chr is not None:
      df = df.loc[df["Chromosome"] == chr, :]

    return pr.PyRanges(df)

def load_encode_sequences(narrowpeak_file, fasta_file, chr_name="chr21"):
    """
    Load DNA sequences from ENCODE narrowPeak regions.

    Args:
        narrowpeak_file (str): Path to .bed.gz file with ChIP-seq peaks
        fasta_file (str): Path to chromosome FASTA file
        chr_name (str): Chromosome name to filter for

    Returns:
        list: DNA sequences from peak regions as strings
    """

    narrow_peak_regions = load_narrow_peak(narrowpeak_file)
    fasta = Fasta(fasta_file)

    sequence_list = []
    # Gather sequences
    for row in narrow_peak_regions.df.itertuples(index=False):
      if getattr(row, "Chromosome") == chr_name:
            start = getattr(row, "Start")
            end = getattr(row, "End")
            seq = str(fasta[chr_name][start:end]).upper() # Our dict is uppercase only
            sequence_list.append(seq)

    return sequence_list

# Load CTCF sequences and run motif discovery
ctcf_sequences = load_encode_sequences("ENCFF029THO.bed.gz", "chr21.fa", "chr21")
print(f"Loaded {len(ctcf_sequences)} CTCF binding sequences from chr21")
print(f"Average sequence length: {np.mean([len(seq) for seq in ctcf_sequences]):.1f} bp")

# Discover CTCF motif (L=14 for computational efficiency)
print("\nDiscovering CTCF motif...")
run_GibbsSampler(ctcf_sequences, 14, 50)

import numpy as np

ref = "CCGCGNGGNGGCAG"
discovered = "CTTTCAAAGGGCGG"

# reverse complement helper
comp = {"A":"T","C":"G","G":"C","T":"A","N":"N"}
def revcomp(s): return "".join(comp.get(b,"N") for b in s[::-1])

def pct_identity(a, b, wildcard=False):
    L = min(len(a), len(b))
    if wildcard:
        match = sum((a[i]==b[i]) or (a[i]=='N') or (b[i]=='N') for i in range(L))
    else:
        match = sum(a[i]==b[i] for i in range(L))
    return 100.0 * match / L

discovered_rc = revcomp(discovered)

print("Ref:      ", ref)
print("Discovered:     ", discovered)
print("Discovered RC:  ", discovered_rc)
print("\nWith 'N' as wildcard % identity vs ref:")
print(" forward:", f"{pct_identity(discovered,   ref, wildcard=True):.1f}")
print(" revcomp:", f"{pct_identity(discovered_rc, ref, wildcard=True):.1f}")
