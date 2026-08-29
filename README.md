# 🚀 daily-arXiv-ai-enhanced

> [!CAUTION]
> 若您所在法域对学术数据有审查要求，谨慎运行本代码；任何二次分发版本必须履行合规审查（包括但不限于原始论文合规性、AI合规性）义务，否则一切法律后果由下游自行承担。

> [!CAUTION]
> If your jurisdiction has censorship requirements for academic data, run this code with caution; any secondary distribution version must remove the entrance accessible to China and fulfill the content review obligations, otherwise all legal consequences will be borne by the downstream.


This innovative tool transforms how you stay updated with arXiv papers by combining automated crawling with AI-powered summarization.


## ✨ Key Features

🎯 **Zero Infrastructure Required**
- Leverages GitHub Actions and Pages - no server needed
- Completely free to deploy and use

🤖 **Smart AI Summarization**
- Daily paper crawling with DeepSeek-powered summaries
- Cost-effective: Only ~0.2 CNY per day

💫 **Smart Reading Experience**
- Personalized paper highlighting based on your interests
- Cross-device compatibility (desktop & mobile)
- Local preference storage for privacy
- Flexible date range filtering

🧩 **SKILL System**
- Plug-and-play skill modules for customizing paper filtering

⚙️ **Easy Preference Export & Integration**
- One-click copy in Settings to export your keywords and authors configuration
- Seamlessly combine exported preferences with SKILL for reproducible and shareable setups

👉 **[Try it now!](https://dw-dengwei.github.io/daily-arXiv-ai-enhanced/)** - No installation required



https://github.com/user-attachments/assets/b25712a4-fb8d-484f-863d-e8da6922f9d7




# How to use
This repo will daily crawl arXiv papers about **cs.CV, cs.GR, cs.CL, cs.AI, cs.CE, cs.GT, cs.IT, cs.LG**, and use **DeepSeek** to summarize the papers in **Chinese**.
If you wish to crawl other arXiv categories, use other LLMs, or other languages, please follow the instructions.
Otherwise, you can watch the video above first and directly use this repo in https://dw-dengwei.github.io/daily-arXiv-ai-enhanced/. Please star it if you like :)

<details>
   <summary> If you want to customize categories, LLMs, or languages, click here.  </summary>

## Instructions
1. Fork this repo to your own account and delete my own information in [buy-me-a-coffee](./buy-me-a-coffee/README.md).
2. Go to: your-own-repo -> Settings -> Secrets and variables -> Actions
3. Go to Secrets. Secrets are encrypted and used for sensitive data
4. Create two repository secrets named `OPENAI_API_KEY` and `OPENAI_BASE_URL`, and input corresponding values.
5. [Optional] Set a password in `secrets.ACCESS_PASSWORD` if you do not wish others to access your page. (see https://github.com/dw-dengwei/daily-arXiv-ai-enhanced/pull/64)
6. Go to Variables. Variables are shown as plain text and are used for non-sensitive data
7. Create the following repository variables:
   1. `CATEGORIES`: separate the categories with ",", such as "cs.CL, cs.CV"
   2. `LANGUAGE`: such as "Chinese" or "English"
   3. `MODEL_NAME`: such as "deepseek-chat"
   4. `EMAIL`: your email for push to GitHub
   5. `NAME`: your name for push to GitHub
8. Go to your-own-repo -> Actions -> arXiv-daily-ai-enhanced
9. You can manually click **Run workflow** to test if it works well (it may take about one hour). By default, this action will automatically run every day. You can modify it in `.github/workflows/run.yml`
10. Set up GitHub pages: Go to your own repo -> Settings -> Pages. In `Build and deployment`, set `Source="Deploy from a branch"`, `Branch="main", "/(root)"`. Wait for a few minutes, go to https://\<username\>.github.io/daily-arXiv-ai-enhanced/. Please see this [issue](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced/issues/14) for more precise instructions.

</details>

## Optional workflow variables

Set these in **Settings -> Secrets and variables -> Actions -> Variables**.

The scheduled workflow runs at `02:15`, `08:15`, `14:15`, and `20:15` UTC on weekdays. Each run has a 5-hour timeout and uses checkpointed crawl metadata and AI results on the `data` branch, so a later run can continue the same UTC day's unfinished work. Scheduled runs resolve the processing date from their cron slot, so a GitHub Actions delay past UTC midnight still resumes the date that the run was scheduled for. Any AI result that is not a complete structured success is deferred and retried by the next scheduled run for that UTC date, while completed papers are preserved. If the raw data, AI-enhanced JSONL, and Markdown file for the day are already complete, the workflow exits early without reprocessing. Historical dates can be resumed manually with the `process_date` workflow input.

| Variable | Default | Usage |
| --- | --- | --- |
| `AI_INPUT_SOURCE` | `abstract` | `abstract` sends only the arXiv metadata abstract to the LLM. `full` also fetches arXiv full text from `arxiv.org/html/{id}` at runtime and sends it in the prompt. Full text is not saved to data files. |
| `AI_MAX_WORKERS` | `1` | Number of concurrent AI enhancement workers. Keep this at `1` for low-tier or rate-limited API plans. |
| `AI_MIN_INTERVAL_SECONDS` | `3` | Minimum delay between AI requests across workers. Increase this if the provider returns 429. |
| `AI_RETRY_ATTEMPTS` | `2` | Retry attempts for each AI request before preserving metadata as a deferred result for the next same-day run. |
| `AI_RETRY_BASE_SECONDS` | `10` | Base delay for exponential AI retry backoff. |
| `AI_RETRY_MAX_SECONDS` | `60` | Maximum delay for AI retry backoff. |
| `AI_REQUEST_TIMEOUT_SECONDS` | `300` | Per-request timeout for the AI provider. |
| `AI_CIRCUIT_BREAKER_FAILURES` | `5` | Stops further AI calls after this many consecutive retryable failures. Account-wide authentication or balance errors stop calls immediately. Set to `0` to disable only the consecutive-failure threshold. |
| `AI_MAX_SECONDS` | `0` | AI-stage time budget. `0` means no application-level time budget. |
| `AI_CHECKPOINT_ENABLED` | `true` | Periodically saves partial AI results to the `data` branch. |
| `AI_CHECKPOINT_INTERVAL_SECONDS` | `600` | Checkpoint interval. The workflow also attempts one final checkpoint when it receives a termination signal. |
| `SENSITIVE_CHECK_ENABLED` | `false` | Enables the optional sensitive-word check. Disabled by default. |

When `AI_INPUT_SOURCE=full`, these additional variables control full-text fetching:

| Variable | Default | Usage |
| --- | --- | --- |
| `AI_FULL_TEXT_MAX_CHARS` | `0` | Maximum full-text characters sent to the model. `0` means no application-level truncation. |
| `AI_FULL_TEXT_TIMEOUT_SECONDS` | `20` | Timeout for fetching arXiv full text. |
| `AI_FULL_TEXT_MIN_INTERVAL_SECONDS` | `0.5` | Minimum delay between arXiv full-text fetches. |
| `AI_FULL_TEXT_RETRY_ATTEMPTS` | `2` | Retry attempts for fetching full text before falling back to abstract. |
| `AI_FULL_TEXT_RETRY_BASE_SECONDS` | `2` | Base delay for full-text fetch retry backoff. |
| `AI_FULL_TEXT_RETRY_MAX_SECONDS` | `30` | Maximum delay for full-text fetch retry backoff. |
| `AI_FULL_TEXT_MIN_CHARS` | `2000` | Minimum extracted full-text length. Shorter content is treated as unavailable and falls back to abstract. |

# Contributors
Thanks to the following special contributors for contributing code, discovering bugs, and sharing useful ideas for this project!!!
<table>
  <tbody>
    <tr>
      <td align="center" valign="top">
        <a href="https://github.com/JianGuanTHU"><img src="https://avatars.githubusercontent.com/u/44895708?v=4" width="100px;" alt="JianGuanTHU"/><br /><sub><b>JianGuanTHU</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/Chi-hong22"><img src="https://avatars.githubusercontent.com/u/75403952?v=4" width="100px;" alt="Chi-hong22"/><br /><sub><b>Chi-hong22</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/chaozg"><img src="https://avatars.githubusercontent.com/u/69794131?v=4" width="100px;" alt="chaozg"/><br /><sub><b>chaozg</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/quantum-ctrl"><img src="https://avatars.githubusercontent.com/u/16505311?v=4" width="100px;" alt="quantum-ctrl"/><br /><sub><b>quantum-ctrl</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/Zhao2z"><img src="https://avatars.githubusercontent.com/u/141019403?v=4" width="100px;" alt="Zhao2z"/><br /><sub><b>Zhao2z</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/eclipse0922"><img src="https://avatars.githubusercontent.com/u/6214316?v=4" width="100px;" alt="eclipse0922"/><br /><sub><b>eclipse0922</b></sub></a><br />
      </td>
    </tr>


  </tbody>
  <tbody>
   <tr>
      <td align="center" valign="top">
        <a href="https://github.com/xuemian168"><img src="https://avatars.githubusercontent.com/u/38741078?v=4" width="100px;" alt="xuemian168"/><br /><sub><b>xuemian168</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/Lrrrr549"><img src="https://avatars.githubusercontent.com/u/71866027?v=4" width="100px;" alt="Lrrrr549"/><br /><sub><b>Lrrrr549</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/AinzRimuru"><img src="https://avatars.githubusercontent.com/u/59441476?v=4" width="100px;" alt="AinzRimuru"/><br /><sub><b>AinzRimuru</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/fengxueguiren"><img src="https://avatars.githubusercontent.com/u/153522370?v=4" width="100px;" alt="fengxueguiren"/><br /><sub><b>fengxueguiren</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://github.com/zerocpp"><img src="https://avatars.githubusercontent.com/u/2630297?v=4" width="100px;" alt="fengxueguiren"/><br /><sub><b>zerocpp</b></sub></a><br />
      </td>
   </tr>
  </tbody>
</table>

# Acknowledgement
We sincerely thank the following individuals and organizations for their promotion and support!!!
<table>
  <tbody>
    <tr>
      <td align="center" valign="top">
        <a href="https://x.com/GitHub_Daily/status/1930610556731318781"><img src="https://pbs.twimg.com/profile_images/1660876795347111937/EIo6fIr4_400x400.jpg" width="100px;" alt="Github_Daily"/><br /><sub><b>Github_Daily</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://x.com/aigclink/status/1930897858963853746"><img src="https://pbs.twimg.com/profile_images/1729450995850027008/gllXr6bh_400x400.jpg" width="100px;" alt="AIGCLINK"/><br /><sub><b>AIGCLINK</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://www.ruanyifeng.com/blog/2025/06/weekly-issue-353.html"><img src="https://avatars.githubusercontent.com/u/905434" width="100px;" alt="阮一峰的网络日志"/><br /><sub><b>阮一峰的网络日志 <br> 科技爱好者周刊 <br> （第 353 期）</b></sub></a><br />
      </td>
      <td align="center" valign="top">
        <a href="https://hellogithub.com/periodical/volume/111"><img src="https://github.com/user-attachments/assets/eff6b6dd-0323-40c4-9db6-444a51bbc80a" width="100px;" alt="《HelloGitHub》第 111 期"/><br /><sub><b>《HelloGitHub》<br> 月刊第 111 期</b></sub></a><br />
      </td>
    </tr>
  </tbody>
</table>


# Star history

[![Stargazers over time](https://starchart.cc/dw-dengwei/daily-arXiv-ai-enhanced.svg?variant=adaptive)](https://starchart.cc/dw-dengwei/daily-arXiv-ai-enhanced)

# Buy me a coffee
[here](./buy-me-a-coffee/README.md)
