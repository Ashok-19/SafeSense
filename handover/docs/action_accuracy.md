# Action accuracy on NTU-120 (cross-subject: test people never seen in training)

Sorted by SafeSense-Pose (682k, `safesense_pose_ntu120.pt`, 78.1% overall). Original SafeSense (496k, `safesense_orig_ntu120.pt`, 74.0% overall) alongside. People: 2 = two-person interaction (the original model sees one person only, so it is weaker there). Demo: in the default demo action list.

| # | Action | SafeSense-Pose | Original SafeSense | People | Demo |
|---|---|---|---|---|---|
| 1 | jump up | 100.0% | 98.2% | 1 | yes |
| 2 | arm circles | 99.8% | 98.4% | 1 | yes |
| 3 | squat down | 99.7% | 98.3% | 1 | yes |
| 4 | cross arms | 99.7% | 98.1% | 1 | yes |
| 5 | arm swings | 99.7% | 98.8% | 1 |  |
| 6 | pickup | 99.6% | 98.9% | 1 | yes |
| 7 | walking apart from each other | 99.3% | 93.5% | 2 |  |
| 8 | walking towards each other | 99.3% | 82.4% | 2 |  |
| 9 | cheers and drink | 99.0% | 92.3% | 2 |  |
| 10 | staggering | 98.9% | 87.7% | 1 | yes |
| 11 | hugging other person | 98.9% | 96.0% | 2 |  |
| 12 | sitting down | 98.5% | 99.6% | 1 | yes |
| 13 | standing up | 98.5% | 99.3% | 1 | yes |
| 14 | falling | 98.2% | 98.2% | 1 | yes |
| 15 | bounce ball | 97.9% | 93.9% | 1 |  |
| 16 | nod head/bow | 97.8% | 98.2% | 1 |  |
| 17 | pushing other person | 97.8% | 86.2% | 2 |  |
| 18 | carry something with other person | 97.6% | 85.2% | 2 |  |
| 19 | move heavy objects | 97.5% | 88.4% | 1 |  |
| 20 | salute | 97.1% | 95.3% | 1 |  |
| 21 | cross hands in front (say stop) | 97.1% | 98.9% | 1 |  |
| 22 | shake head | 97.1% | 91.3% | 1 |  |
| 23 | support somebody with hand | 97.0% | 94.1% | 2 |  |
| 24 | high-five | 96.5% | 84.4% | 2 |  |
| 25 | kicking something | 96.4% | 94.6% | 1 | yes |
| 26 | wear jacket | 96.4% | 98.2% | 1 |  |
| 27 | finger-guessing game (playing rock-paper-scissors) | 96.4% | 88.9% | 2 |  |
| 28 | check time (from watch) | 96.0% | 85.9% | 1 |  |
| 29 | butt kicks (kick backward) | 96.0% | 92.2% | 1 |  |
| 30 | hopping (one foot jumping) | 95.6% | 96.0% | 1 |  |
| 31 | whisper in other person's ear | 95.5% | 72.4% | 2 |  |
| 32 | take off jacket | 95.3% | 96.0% | 1 |  |
| 33 | touch back (backache) | 95.3% | 96.4% | 1 |  |
| 34 | point finger at the other person | 94.9% | 54.7% | 2 |  |
| 35 | knock over other person (hit with body) | 94.6% | 64.8% | 2 |  |
| 36 | running on the spot | 94.4% | 92.7% | 1 | yes |
| 37 | hands up (both hands) | 94.2% | 96.5% | 1 | yes |
| 38 | step on foot | 94.1% | 74.6% | 2 |  |
| 39 | follow other person | 93.8% | 89.9% | 2 |  |
| 40 | put the palms together | 93.5% | 87.3% | 1 |  |
| 41 | put on headphone | 93.4% | 90.9% | 1 |  |
| 42 | thumb down | 93.2% | 84.7% | 1 |  |
| 43 | put on bag | 93.2% | 90.1% | 1 |  |
| 44 | side kick | 92.5% | 89.0% | 1 |  |
| 45 | touch chest (stomachache/heart pain) | 92.4% | 87.3% | 1 | yes |
| 46 | kicking other person | 92.0% | 67.4% | 2 |  |
| 47 | handshaking | 91.7% | 81.2% | 2 |  |
| 48 | cross toe touch | 90.8% | 87.3% | 1 |  |
| 49 | take off glasses | 90.5% | 88.7% | 1 |  |
| 50 | take off a hat/cap | 90.5% | 93.4% | 1 |  |
| 51 | wear on glasses | 90.1% | 93.4% | 1 |  |
| 52 | juggling table tennis balls | 89.9% | 79.9% | 1 |  |
| 53 | touch neck (neckache) | 89.9% | 84.1% | 1 |  |
| 54 | exchange things with other person | 89.6% | 73.9% | 2 |  |
| 55 | take off headphone | 89.4% | 81.5% | 1 |  |
| 56 | drop | 89.1% | 76.0% | 1 |  |
| 57 | throw | 88.0% | 77.8% | 1 |  |
| 58 | grab other person's stuff | 87.8% | 56.2% | 2 |  |
| 59 | hush (quite) | 87.8% | 74.2% | 1 |  |
| 60 | tear up paper | 87.5% | 77.9% | 1 |  |
| 61 | toss a coin | 87.4% | 79.9% | 1 |  |
| 62 | take something out of a bag | 87.3% | 68.2% | 1 |  |
| 63 | take off bag | 87.2% | 84.9% | 1 |  |
| 64 | make a phone call/answer phone | 86.9% | 86.9% | 1 | yes |
| 65 | stretch oneself | 86.8% | 84.2% | 1 |  |
| 66 | nausea or vomiting condition | 85.5% | 86.2% | 1 |  |
| 67 | put on a hat/cap | 85.3% | 89.3% | 1 |  |
| 68 | sniff (smell) | 84.9% | 79.5% | 1 |  |
| 69 | punching/slapping other person | 84.7% | 61.0% | 2 |  |
| 70 | pat on back of other person | 84.4% | 75.0% | 2 |  |
| 71 | take off a shoe | 84.3% | 79.2% | 1 |  |
| 72 | cheer up | 83.6% | 86.9% | 1 |  |
| 73 | touch head (headache) | 83.0% | 84.8% | 1 | yes |
| 74 | apply cream on face | 82.1% | 84.5% | 1 |  |
| 75 | take a photo of other person | 81.1% | 67.7% | 2 |  |
| 76 | reach into pocket | 78.1% | 83.2% | 1 |  |
| 77 | giving something to other person | 77.5% | 65.9% | 2 |  |
| 78 | hand waving | 77.0% | 81.8% | 1 | yes |
| 79 | wipe face | 76.4% | 81.5% | 1 |  |
| 80 | drink water | 75.5% | 80.3% | 1 | yes |
| 81 | shoot at the basket | 75.5% | 83.9% | 1 |  |
| 82 | touch other person's pocket | 74.9% | 69.1% | 2 |  |
| 83 | clapping | 74.7% | 81.0% | 1 | yes |
| 84 | pointing to something with finger | 74.6% | 71.7% | 1 |  |
| 85 | tennis bat swing | 74.6% | 72.1% | 1 |  |
| 86 | eat meal/snack | 74.6% | 76.0% | 1 |  |
| 87 | cutting nails | 74.3% | 53.1% | 1 |  |
| 88 | wear a shoe | 72.9% | 73.3% | 1 |  |
| 89 | flick hair | 72.9% | 63.6% | 1 |  |
| 90 | sneeze/cough | 72.1% | 78.6% | 1 |  |
| 91 | open bottle | 70.5% | 62.3% | 1 |  |
| 92 | brushing hair | 68.9% | 79.1% | 1 |  |
| 93 | yawn | 68.3% | 72.7% | 1 |  |
| 94 | brushing teeth | 67.8% | 80.2% | 1 |  |
| 95 | throw up cap/hat | 64.9% | 68.2% | 1 |  |
| 96 | put something into a bag | 64.5% | 65.7% | 1 |  |
| 97 | taking a selfie | 64.5% | 83.3% | 1 |  |
| 98 | shake fist | 62.0% | 53.1% | 1 |  |
| 99 | blow nose | 61.9% | 64.3% | 1 |  |
| 100 | shoot at other person with a gun | 60.4% | 32.5% | 2 |  |
| 101 | thumb up | 56.4% | 47.5% | 1 |  |
| 102 | rub two hands together | 56.2% | 67.8% | 1 |  |
| 103 | use a fan (with hand or paper)/feeling warm | 56.0% | 73.5% | 1 |  |
| 104 | snapping fingers | 55.9% | 52.8% | 1 |  |
| 105 | open a box | 53.8% | 50.0% | 1 |  |
| 106 | apply cream on hand back | 51.4% | 58.5% | 1 |  |
| 107 | hit other person with something | 50.6% | 34.6% | 2 |  |
| 108 | typing on a keyboard | 49.1% | 62.2% | 1 |  |
| 109 | make ok sign | 48.5% | 36.0% | 1 |  |
| 110 | ball up paper | 44.7% | 60.7% | 1 |  |
| 111 | wield knife towards other person | 42.0% | 25.0% | 2 |  |
| 112 | staple book | 41.0% | 25.4% | 1 |  |
| 113 | writing | 30.5% | 40.4% | 1 |  |
| 114 | fold paper | 29.6% | 31.1% | 1 |  |
| 115 | reading | 28.2% | 46.5% | 1 |  |
| 116 | counting money | 24.7% | 34.4% | 1 |  |
| 117 | make victory sign | 23.0% | 33.2% | 1 |  |
| 118 | cutting paper (using scissors) | 17.8% | 53.9% | 1 |  |
| 119 | play magic cube | 8.6% | 20.1% | 1 |  |
| 120 | playing with phone/tablet | 8.4% | 21.8% | 1 |  |
