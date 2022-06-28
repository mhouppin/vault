
## Vault

Vault is a UCI-compliant chess engine derivating from Stash, using a neural
network for evaluating positions. Preferably used with a GUI like Arena,
CuteChess, Fritz, etc.

Note that the current project version is not "stable", as I'm still working on
a better Makefile configuration for faster builds.

## Files

The repository consists of the following files:

  * Readme.md, the file you are currently reading.
  * LICENSE, a text file containing the GNU General Public License version 3.
  * src, the directory containing all the source code + a Makefile that can be
    used to compile Vault on Unix-like systems (or Windows if you installed
    MinGW). Note that Git LFS is needed for downloading the network from CLI.

## UCI Parameters

Vault supports for now all these UCI options:

  * #### EvalFile
    Indicates the path of the neural network to use (defaults to <empty>, which
    falls back on using the HCE).

  * #### Threads
    Sets the number of cores used for searching a position (defaults to 1).

  * #### Hash
    Sets the hash table size in MB (defaults to 16).

  * #### Clear Hash
    Clears the hash table.

  * #### MultiPV
    Output the best N lines (principal variations) when searching.
    Leave at 1 for best performance.

  * #### Move Overhead
    Assumes a time delay of x milliseconds due to network and GUI overheads.
    Increase it if the engine often loses games on time. The default value
    of 100 ms should be sufficient for all chess GUIs.
