# Publishing on GitHub and archiving with Zenodo

## GitHub website

1. Extract the ZIP. Open the `terzan5-sonification` folder and check that it contains `README.md`, `requirements.txt`, `scripts/`, `docs/`, and `.gitignore`. On macOS press **Command+Shift+Period** in Finder to show `.gitignore`.
2. In GitHub, choose **+ → New repository**. Enter a name such as `terzan5-sonification`. Choose **Private** while reviewing the upload, or **Public** when ready to share. Leave the README, .gitignore template, and license options unselected for the initial upload.
3. Create the repository. On the empty repository page select **uploading an existing file**. Drag the **contents** of the extracted repository folder into the upload area, preserving `scripts/` and `docs/`. The README should be at the repository root. If the browser skips `.gitignore`, use the terminal method below.
4. Review the list before committing. It should contain the scripts, documentation, requirements, and ignore file. Keep observational inputs, timing tables, generated audio, and conversion logs in local folders until their distribution terms are clear.
5. Use a commit message such as **Add Terzan 5 sonification workflow** and commit to `main`. Open the repository and check that the README renders and all scripts and docs are in the intended subfolders.

[GitHub’s file upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## Terminal alternative

Create an empty GitHub repository as above. Authenticate first, then open a terminal in the extracted `terzan5-sonification` directory. Replace `YOUR-USERNAME` and the repository name with the values from GitHub.

```sh
git init -b main
git add .gitignore README.md requirements.txt docs scripts
git diff --cached --name-only
git commit -m "Add Terzan 5 sonification workflow"
git remote add origin https://github.com/YOUR-USERNAME/terzan5-sonification.git
git push -u origin main
```

If Git requests your identity, configure `user.name` and `user.email` for this repository and retry the commit. GitHub accepts HTTPS authentication through a credential manager or the GitHub CLI.

[GitHub’s instructions for pushing local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

## DOI for a paper

When the code is ready for publication, decide on the license for material you can license. This package makes no license selection. For a Zenodo-linked workflow, enable the public repository in Zenodo’s GitHub settings, publish an immutable-by-convention GitHub release tag such as `v1.0.0`, and verify that Zenodo archived the intended files. Use the DOI for that specific version in the paper. You can also deposit the release ZIP directly in Zenodo.

Suggested reference format after deposit:

> M. Kang, “Terzan 5 Sonification,” software, version 1.0.0, Zenodo, [year]. doi: [version-specific DOI].

[GitHub’s guide to DOI archiving with Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
