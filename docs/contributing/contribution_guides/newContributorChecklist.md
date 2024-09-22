# New Contributor Guide

If you're a new contributor to InvokeAI or Open Source Projects, this is the guide for you.

## New Contributor Checklist

- [x] Set up your local development environment & fork of InvokAI by following [the steps outlined here](../dev-environment.md)
- [x] Set up your local tooling with [this guide](InvokeAI/contributing/LOCAL_DEVELOPMENT/#developing-invokeai-in-vscode). Feel free to skip this step if you already have tooling you're comfortable with.
- [x] Familiarize yourself with [Git](https://www.atlassian.com/git) & our project structure by reading through the [development documentation](development.md)
- [x] Join the [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) channel of the Discord
- [x] Choose an issue to work on! This can be achieved by asking in the #dev-chat channel, tackling a [good first issue](https://github.com/invoke-ai/InvokeAI/contribute) or finding an item on the [roadmap](https://github.com/orgs/invoke-ai/projects/7). If nothing in any of those places catches your eye, feel free to work on something of interest to you!
- [x] Make your first Pull Request with the guide below
- [x] Happy development! Don't be afraid to ask for help - we're happy to help you contribute!

## How do I make a contribution?

Never made an open source contribution before? Wondering how contributions work in our project? Here's a quick rundown!

Before starting these steps, ensure you have your local environment [configured for development](../LOCAL_DEVELOPMENT.md).

1. Find a [good first issue](https://github.com/invoke-ai/InvokeAI/contribute) that you are interested in addressing or a feature that you would like to add. Then, reach out to our team in the [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) channel of the Discord to ensure you are setup for success.
2. Fork the [InvokeAI](https://github.com/invoke-ai/InvokeAI) repository to your GitHub profile. This means that you will have a copy of the repository under **your-GitHub-username/InvokeAI**.
3. Clone the repository to your local machine using:

   ```bash
   git clone https://github.com/your-GitHub-username/InvokeAI.git
   ```

If you're unfamiliar with using Git through the commandline, [GitHub Desktop](https://desktop.github.com) is a easy-to-use alternative with a UI. You can do all the same steps listed here, but through the interface. 4. Create a new branch for your fix using:

    ```bash
    git checkout -b branch-name-here
    ```

5. Make the appropriate changes for the issue you are trying to address or the feature that you want to add.
6. Add the file contents of the changed files to the "snapshot" git uses to manage the state of the project, also known as the index:

    ```bash
    git add -A
    ```

7. Store the contents of the index with a descriptive message.

    ```bash
    git commit -m "Insert a short message of the changes made here"
    ```

8. Push the changes to the remote repository using

    ```bash
    git push origin branch-name-here
    ```

9. Submit a pull request to the **main** branch of the InvokeAI repository. If you're not sure how to, [follow this guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
10. Title the pull request with a short description of the changes made and the issue or bug number associated with your change. For example, you can title an issue like so "Added more log outputting to resolve #1234".
11. In the description of the pull request, explain the changes that you made, any issues you think exist with the pull request you made, and any questions you have for the maintainer. It's OK if your pull request is not perfect (no pull request is), the reviewer will be able to help you fix any problems and improve it!
12. Wait for the pull request to be reviewed by other collaborators.
13. Make changes to the pull request if the reviewer(s) recommend them.
14. Celebrate your success after your pull request is merged!

If you’d like to learn more about contributing to Open Source projects, here is a [Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github).

## Best Practices

- Keep your pull requests small. Smaller pull requests are more likely to be accepted and merged

- Comments! Commenting your code helps reviewers easily understand your contribution
- Use Python and Typescript’s typing systems, and consider using an editor with [LSP](https://microsoft.github.io/language-server-protocol/) support to streamline development
- Make all communications public. This ensure knowledge is shared with the whole community

## **Where can I go for help?**

If you need help, you can ask questions in the [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) channel of the Discord.

For frontend related work, **@pyschedelicious** is the best person to reach out to.

For backend related work, please reach out to **@blessedcoolant**, **@lstein**, **@StAlKeR7779** or **@pyschedelicious**.
