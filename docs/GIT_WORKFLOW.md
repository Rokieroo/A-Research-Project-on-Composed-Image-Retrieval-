# Git workflow and how to use this repository

A beginner-friendly guide to working on this project with Git and GitHub.

## 1. One-time setup

Install [Git](https://git-scm.com/downloads) and Python 3.10 or newer, then tell
Git who you are:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"   # use the email on your GitHub account
```

Clone the repository and set up the Python environment:

```bash
git clone https://github.com/rokieroo/a-research-project-on-composed-image-retrieval-.git cir-project
cd cir-project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                             # should print "5 passed"
```

Download the data (about 13.6 GB, only needed once). See `data/README.md`
for details.

```bash
bash scripts/download_data.sh
```

## 2. The key ideas in 30 seconds

| Term | Meaning |
|---|---|
| **commit** | A saved snapshot of your files with a message saying what changed. |
| **branch** | A separate line of work. `main` is the clean, working version. |
| **push / pull** | Send your commits to GitHub / bring GitHub's commits to your computer. |
| **pull request (PR)** | A request on GitHub to merge a branch into `main`, where you can review the changes first. |

**The golden rule:** `main` should always work. Do new work on a branch, then
merge it through a pull request.

## 3. Everyday workflow

### Step 1: start from an up-to-date `main`

```bash
git switch main
git pull
```

### Step 2: create a branch for one piece of work

```bash
git switch -c feature/cirr-loader
```

Branch name prefixes:

| Prefix | Use for | Example |
|---|---|---|
| `feature/` | new code | `feature/cirr-loader` |
| `exp/` | an experiment | `exp/clip-vitl14-zeroshot` |
| `fix/` | a bug fix | `fix/recall-off-by-one` |
| `docs/` | notes, report | `docs/pic2word-notes` |

### Step 3: make changes and commit often

```bash
git status                 # what changed?
git diff                   # see the exact changes
git add src/datasets/cirr.py tests/test_cirr.py
git commit -m "Add CIRR validation loader"
```

Good commit messages are short, in the imperative mood, and say *what* changed:

- Good: `Add Recall_subset@K for CIRR`, `Fix image path in FashionIQ loader`
- Bad: `update`, `stuff`, `final version 2`

Before you commit, check that you are **not** adding data, checkpoints or
outputs. `.gitignore` already blocks most of these, so if `git status` shows a
large file, stop and find out why.

### Step 4: push the branch to GitHub

```bash
git push -u origin feature/cirr-loader     # first push of a branch
git push                                   # later pushes
```

### Step 5: open a pull request

1. On the repository page, GitHub shows a **Compare & pull request** button.
   Click it.
2. Set the base to `main` and the compare branch to yours.
3. Write what you changed, why, and how you tested it (for example, "`pytest`
   passes, R@1 on CIRR val = 23.4").
4. Read through the **Files changed** tab yourself. It is the best place to
   catch mistakes.
5. Click **Merge pull request**, then **Delete branch**.

### Step 6: update your computer

```bash
git switch main
git pull
git branch -d feature/cirr-loader
```

## 4. Running experiments the reproducible way

1. Copy a config: `cp configs/baseline_clip.yaml configs/my_experiment.yaml`,
   then edit it and give it a new `name`.
2. Commit the config **before** running, so the result matches a commit.
3. Run it:
   ```bash
   python -m src.evaluate --config configs/my_experiment.yaml
   ```
4. Add a row to the results table in `experiments/README.md` with the date,
   config name, commit hash (`git rev-parse --short HEAD`) and scores.
5. Commit the updated table.

Outputs go to `outputs/` and weights to `checkpoints/`. Neither is committed.
Upload weights you want to keep to Hugging Face Hub or Google Drive and link
them in `checkpoints/README.md`.

## 5. Where things go

| What | Where |
|---|---|
| Notes on a paper | `docs/reading_notes/<year>_<author>_<title>.md` (copy `_template.md`) |
| SC203 report and figures | `docs/report/`, `docs/report/figures/` |
| Exploration and plots | `notebooks/` |
| Code used for reported numbers | `src/` |
| Experiment settings | `configs/` |
| Results table | `experiments/README.md` |
| Tests | `tests/` (run with `pytest`) |

## 6. Fixing common problems

| Problem | Fix |
|---|---|
| `git push` is rejected ("fetch first") | `git pull --rebase`, then `git push` again |
| Wrong commit message on your last commit (not pushed yet) | `git commit --amend -m "Better message"` |
| Want to undo changes to a file you haven't committed | `git restore path/to/file` |
| Staged a file by mistake | `git restore --staged path/to/file` |
| Committed a huge file | Don't push. Remove it with `git rm --cached <file>`, add it to `.gitignore`, then `git commit --amend` |
| Merge conflict | Open the file, keep the right parts between the `<<<<<<<` and `>>>>>>>` markers, delete the markers, then `git add <file>` and `git commit` (or `git rebase --continue`) |
| Not sure what state you're in | `git status` always tells you, and usually suggests the next command |

Avoid `git push --force` on `main`. It overwrites history on GitHub.

## 7. Cheat sheet

```bash
git switch main && git pull           # get the latest
git switch -c feature/name            # new branch
git status / git diff                 # inspect
git add <files> && git commit -m "…"  # save a snapshot
git push -u origin feature/name       # upload, then open a PR on GitHub
git log --oneline --graph             # history
```
