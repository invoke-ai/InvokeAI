# Development

## **What do I need to know to help?**

If you are looking to help to with a code contribution, InvokeAI uses several different technologies under the hood: Python (Pydantic, FastAPI, diffusers) and Typescript (React, Redux Toolkit, ChakraUI, Mantine, Konva). Familiarity with StableDiffusion and image generation concepts is helpful, but not essential. 

For more information, please review our area specific documentation:

* #### [InvokeAI Architecure](../ARCHITECTURE.md)
* #### [Frontend Documentation](development_guides/contributingToFrontend.md)
* #### [Node Documentation](../INVOCATIONS.md)
* #### [Local Development](../LOCAL_DEVELOPMENT.md)

If you don't feel ready to make a code contribution yet, no problem! You can also help out in other ways, such as [documentation](documentation.md) or [translation](translation.md).

There are two paths to making a development contribution: 

1. Choosing an open issue to address. Open issues can be found in the [Issues](https://github.com/invoke-ai/InvokeAI/issues?q=is%3Aissue+is%3Aopen) section of the InvokeAI repository. These are tagged by the issue type (bug, enhancement, etc.) along with the “good first issues” tag denoting if they are suitable for first time contributors.
    1. Additional items can be found on our [roadmap](https://github.com/orgs/invoke-ai/projects/7). The roadmap is organized in terms of priority, and contains features of varying size and complexity. If there is an inflight item you’d like to help with, reach out to the contributor assigned to the item to see how you can help. 
2. Opening a new issue or feature to add. **Please make sure you have searched through existing issues before creating new ones.**

*Regardless of what you choose, please post in the  [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) channel of the Discord before you start development in order to confirm that the issue or feature is aligned with the current direction of the project. We value our contributors time and effort and want to ensure that no one’s time is being misspent.*

## Best Practices: 
* Keep your pull requests small. Smaller pull requests are more likely to be accepted and merged
* Comments! Commenting your code helps reviwers easily understand your contribution
* Use Python and Typescript’s typing systems, and consider using an editor with [LSP](https://microsoft.github.io/language-server-protocol/) support to streamline development
* Make all communications public. This ensure knowledge is shared with the whole community

## **How do I make a contribution?**

Never made an open source contribution before? Wondering how contributions work in our project? Here's a quick rundown!

Before starting these steps, ensure you have your local environment [configured for development](../LOCAL_DEVELOPMENT.md).

1.  Find a [good first issue](https://github.com/invoke-ai/InvokeAI/contribute) that you are interested in addressing or a feature that you would like to add. Then, reach out to our team in the [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) channel of the Discord to ensure you are  setup for success. 
2. Fork the [InvokeAI](https://github.com/invoke-ai/InvokeAI) repository to your GitHub profile. This means that you will have a copy of the repository under **your-GitHub-username/InvokeAI**.
3. Clone the repository to your local machine using:

```bash
git clone https://github.com/your-GitHub-username/InvokeAI.git
```

If you're unfamiliar with using Git through the commandline, [GitHub Desktop](https://desktop.github.com) is a easy-to-use alternative with a UI. You can do all the same steps listed here, but through the interface. 

4. Create a new branch for your fix using:

```bash
git checkout -b branch-name-here
```

5. Make the appropriate changes for the issue you are trying to address or the feature that you want to add.
6. Add the file contents of the changed files to the "snapshot" git uses to manage the state of the project, also known as the index:

```bash
git add insert-paths-of-changed-files-here
```

7. Store the contents of the index with a descriptive message.

```bash
git commit -m "Insert a short message of the changes made here"
```

8. Push the changes to the remote repository using

```markdown
git push origin branch-name-here
```

9. Submit a pull request to the **main** branch of the InvokeAI repository.
10. Title the pull request with a short description of the changes made and the issue or bug number associated with your change. For example, you can title an issue like so "Added more log outputting to resolve #1234".
11. In the description of the pull request, explain the changes that you made, any issues you think exist with the pull request you made, and any questions you have for the maintainer. It's OK if your pull request is not perfect (no pull request is), the reviewer will be able to help you fix any problems and improve it!
12. Wait for the pull request to be reviewed by other collaborators.
13. Make changes to the pull request if the reviewer(s) recommend them.
14. Celebrate your success after your pull request is merged!

If you’d like to learn more about contributing to Open Source projects, here is a [Getting Started Guide](https://opensource.com/article/19/7/create-pull-request-github). 

## **Where can I go for help?**

If you need help, you can ask questions in the [#dev-chat](https://discord.com/channels/1020123559063990373/1049495067846524939) channel of the Discord.

For frontend related work, **@pyschedelicious** is the best person to reach out to. 

For backend related work, please reach out to **@blessedcoolant**, **@lstein**, **@StAlKeR7779** or **@pyschedelicious**.

## **What does the Code of Conduct mean for me?**

Our [Code of Conduct](CODE_OF_CONDUCT.md)  means that you are responsible for treating everyone on the project with respect and courtesy regardless of their identity. If you are the victim of any inappropriate behavior or comments as described in our Code of Conduct, we are here for you and will do the best to ensure that the abuser is reprimanded appropriately, per our code.

