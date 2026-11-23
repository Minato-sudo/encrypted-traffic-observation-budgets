# Phase 6: recording and upload guide

What the brief requires, and how to produce it on this machine.

| Requirement | How it is met |
|---|---|
| Screen recording with voice-over | OBS Studio, already installed at `/usr/bin/obs` |
| Both members actively participate | One contiguous half each, slides 1&ndash;6 and 7&ndash;14, with both voices on slide 1 and slide 14 and the handoff spoken aloud at slide 6 |
| Different parts explained by each | Talha: the problem, the gap, the objectives, the methodology, the implementation and the live demo. Zain: every result, the comparison with published work, the findings, the limitations and the conclusion |
| Problem, objectives, methodology, implementation, results, key findings, conclusion | Slides 2&ndash;3, 4&ndash;5, 6, 7&ndash;12, 13, 14 |
| Demonstrate implementation, code, experiments, results | Slide 6, per `DEMO_RUNSHEET.md`, live terminal, driven by Talha |
| Clear audio, readable recording, professional quality | Settings below |
| Upload to YouTube or Vimeo and share the link | Steps at the end |

---

## Check this first: the microphone is not exposed yet

The machine has a working built-in microphone, an ALC257 on the onboard Intel
card, and ALSA can see it:

```bash
arecord -l
```

```
card 0: PCH [HDA Intel PCH], device 0: ALC257 Analog
```

But PulseAudio has that card in an **output-only** profile, so no capture source
is published and OBS has nothing to record. Check:

```bash
pactl list short sources | grep -v monitor
```

If that prints nothing, switch the card to the duplex profile, which the card
reports as available:

```bash
pactl set-card-profile alsa_card.pci-0000_00_1f.3 output:analog-stereo+input:analog-stereo
```

Then re-run the `pactl list short sources` check above. A real source should
appear, and it is the one to select in OBS. The change is per-user and
reversible: `output:analog-stereo` puts it back.

Two further notes:

- The external 3.5 mm microphone jack reports **not available**, meaning nothing
  is plugged into it. The internal microphone is what you will get.
- **A wired headset microphone is still worth using.** The internal mic sits in
  the chassis and picks up fan noise, which becomes obvious under a continuous
  15-minute narration. Plug one in and the jack port becomes available, taking
  priority over the internal mic.

Do not record voice separately on a phone unless you are willing to align it by
hand afterwards.

---

## Displaying the deck

Open `Phase6_Presentation.pdf` in Okular and press **Ctrl+Shift+P** for
presentation mode, which gives one slide per screen with no window chrome.

```bash
okular slides/Phase6_Presentation.pdf
```

The deck is authored at exactly 13.333 by 7.5 inches, which is 16:9, so it
fills a 1920 by 1080 capture with no letterboxing. Advance with the right arrow
or space, and **stop at slide 14**: pages 15 and 16 are backup slides held for
questions.

Set the display resolution to **1920 by 1080** before recording, even if the
monitor is larger. Recording a 2560-wide screen and letting YouTube downscale it
is what makes terminal text turn to mush.

---

## OBS settings

**Settings, Output, Recording:**

| Setting | Value |
|---|---|
| Recording format | `mkv` (survives a crash; remux to mp4 afterwards) |
| Encoder | x264, or NVENC if the GPU offers it |
| Rate control | CBR |
| Bitrate | 12000 Kbps |
| Keyframe interval | 2 s |
| CPU preset | `veryfast` |
| Profile | high |

**Settings, Video:** base and output resolution both **1920x1080**, **30 fps**.
Sixty frames per second doubles the file for a screen recording of static
slides and buys nothing.

**Settings, Audio:** sample rate 48 kHz, stereo.

**Scene, two sources:**

1. **Screen Capture (XSHM)** for the whole display. The session here is X11, so
   XSHM capture works; on Wayland you would need PipeWire capture instead.
2. **Audio Input Capture** set to the microphone you attached, **not** a
   monitor device.

On the microphone source add two filters, in this order:

- **Noise Suppression** (RNNoise). Removes fan and room hum.
- **Compressor**, ratio 4:1, threshold around -18 dB. Levels out the difference
  between the two speakers, which otherwise makes one member sound distant.

Watch the audio meter while doing a test read of slide 1. Peaks should sit
around **-12 dB** and never touch 0. If the meter barely moves, OBS is on the
wrong source.

---

## Recording in segments

Do not attempt one unbroken take. Each member's slide ranges are contiguous, so
record six files:

| File | Speaker | Slides | Length |
|---|---|---|---|
| `part1_talha.mkv` | Talha, with Zain's one line on slide 1 | 1&ndash;5 | 5:30 |
| `part2_talha_demo.mkv` | Talha, at the terminal | 6 | 1:57 |
| `part3_zain.mkv` | Zain | 7&ndash;12 | 5:39 |
| `part4_zain_close.mkv` | Zain, with Talha on slide 14 | 13&ndash;14 | 2:25 |

Four files rather than one long take each, because a 5-minute unbroken read is
hard to get clean and a fluffed sentence 4 minutes in costs the whole take.

Two of them need both members at the microphone: `part1` for Zain's
self-introduction on slide 1, and `part4` for Talha's closing line on slide 14.
Everything else is one voice.

The demonstration is its own file because it is the only segment that leaves
the slides for the terminal. Record it last, once the commands are familiar.

Start each file with two seconds of silence on the first slide and end with two
seconds after the last word. That gives clean joins.

---

## Joining the parts

No video editor is installed here, but the parts are all encoded identically, so
`ffmpeg` can concatenate them without re-encoding:

```bash
cd ~/Videos
printf "file '%s'\n" part1_talha.mkv part2_talha_demo.mkv part3_zain.mkv \
    part4_zain_close.mkv > parts.txt
ffmpeg -f concat -safe 0 -i parts.txt -c copy Phase6_Presentation_i232548_i232582.mkv
```

Then remux to mp4 for upload, again without re-encoding:

```bash
ffmpeg -i Phase6_Presentation_i232548_i232582.mkv -c copy \
       Phase6_Presentation_i232548_i232582.mp4
```

Check the result end to end before uploading, with particular attention to the
joins and to the audio level across the speaker changes:

```bash
ffprobe -hide_banner Phase6_Presentation_i232548_i232582.mp4
```

If a segment needs trimming, cut on a slide boundary:

```bash
ffmpeg -i part3_zain.mkv -ss 00:00:03 -c copy part3_trimmed.mkv
```

---

## Before you upload

- [ ] Both voices are clearly audible and at a similar level.
- [ ] Both members are named out loud, with roll numbers, in the first minute
      and again in the last. With a single handoff this is what proves the
      two-member requirement to anyone who samples the video rather than
      watching it through.
- [ ] The handoff at the end of slide 6 is spoken, not a silent cut.
- [ ] The terminal text is readable when the video is played at 1080p in a
      window, not only full screen.
- [ ] No notification, message preview or personal file name appears anywhere.
- [ ] No number spoken contradicts the slide it is spoken over.
- [ ] The demo shows code, an experiment result and the generated report, which
      is the brief's "implementation, code, experiments, and results".

---

## Uploading

YouTube is the simpler of the two options the brief allows.

1. Go to <https://studio.youtube.com>, **Create**, **Upload videos**.
2. **Title:**
   `Observation Budgets and Padding Effects in Raw-Byte Network Traffic Classification | CN Research Project | FAST-NUCES`
3. **Description:**

   ```
   Computer Networks research project, Fall 2026, FAST-NUCES Islamabad.
   Topic 15: Observation Budgets and Padding Effects in Raw-Byte Network
   Traffic Classification.

   Muhammad Talha Arshad (i232548)
   Zain Shahid (i232582)
   BS Data Science

   We treat the observation budget, the number of bytes a classifier reads per
   flow, as a quantity to be measured rather than inherited. Macro-F1 saturates
   at 512 bytes on USTC-TFC2016 and 256 bytes on ISCX VPN-nonVPN; every budget
   used by twelve recent studies lies at or past that point. A causal
   early-exit classifier matches the best fixed-depth head at 28 percent of the
   byte cost.

   Code, result data and the paper:
   https://github.com/Minato-sudo/encrypted-traffic-observation-budgets

   00:00 Introduction
   00:26 The problem, and the four gaps
   01:36 Padding is confounded with budget; objectives
   02:50 Methodology: the pipeline and the leakage gate
   03:56 Padding decomposed, and the causal classifier
   05:30 Implementation and live demonstration
   07:28 Results: the gate, and the budget response
   08:58 Results: the architecture control and per-class failure
   10:12 Results: padding attribution
   10:46 Results: accuracy per byte read
   11:34 Results: the train/test split confound
   12:17 Comparison with the base papers
   13:07 Key findings and limitations
   14:36 Conclusion and thank you
   ```

   **Fix those timestamps against the finished video before posting.** The ones
   above are derived from the script's word counts, not from a real recording.
   Timestamps that do not match are worse than none at all.
4. **Visibility: Unlisted.** Anyone with the link can watch, which is what the
   submission needs, and the video does not appear in search or on the channel.
   Do not use Private: a private video is invisible to the instructor unless
   their Google account is added individually.
5. Set **"No, it's not made for kids"**.
6. After processing finishes, **open the link in a private browsing window** to
   confirm it plays without being signed in. This is the step most often
   skipped, and a link that only works for the uploader reads as no submission
   at all.
7. Submit the link, and keep the local `.mp4` until the marks are released.

---

## Vimeo, if you prefer it

Free Vimeo accounts cap uploads at 500 MB per week, which a 15-minute 1080p
recording at 12 Mbps will exceed, at roughly 1.3 GB. Either re-encode smaller first:

```bash
ffmpeg -i Phase6_Presentation_i232548_i232582.mp4 -c:v libx264 -crf 26 \
       -preset slow -c:a aac -b:a 128k Phase6_small.mp4
```

or use YouTube, which has no such limit.
