import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = {filename: (1-damping_factor)/len(corpus)
              if len(corpus[page]) != 0 else 1 / len(corpus) for filename in corpus}
    for filename in corpus[page]:
        result[filename] += damping_factor / len(corpus[page])
    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    seq = [filename for filename in corpus]
    pages = []
    result = {filename: 0 for filename in corpus}
    for i in range(n):
        if i == 0:
            page = random.choice(seq)
        else:
            prob_dis = transition_model(corpus, pages[i - 1], damping_factor)
            page = random.choices(
                seq, [prob for _, prob in prob_dis.items()])[0]
        pages.append(page)
        result[page] += 1 / n
    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    from copy import deepcopy

    threshold = 0.001
    N = len(corpus)
    converged = False

    result = {page: 1 / N for page in corpus}

    while not converged:
        prev_result = deepcopy(result)

        for page in result:
            result[page] = (1 - damping_factor) / N + damping_factor * \
                sum(PR_i_divide_by_NumLinks_i(corpus, prev_result, page))
            converged = abs(result[page] - prev_result[page]) < threshold

    return result


def PR_i_divide_by_NumLinks_i(corpus, prev_result, page):
    page_ranks = []
    for page_i in corpus:
        if len(corpus[page_i]) == 0:
            page_ranks.append(prev_result[page_i] / len(corpus))
        elif page in corpus[page_i]:
            page_ranks.append(prev_result[page_i] / len(corpus[page_i]))

    return page_ranks


if __name__ == "__main__":
    main()
