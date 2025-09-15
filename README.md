# Diffusion-based superresolution 
 
This repository contains a from-scratch implementation of [NanoGPT (decoder-only)](https://github.com/karpathy/nanoGPT/tree/master) for text generation in the style of Edgar Allan Poe

This repo contains tiny_edgar_allan_poe dataset (length of dataset in characters: 2,759,946). 

https://huggingface.co/datasets/Cagatayisil/tiny_edgar_allan_poe

The Language model here uses a character-level tokenizer.

Trained on Macbook M2 for 100,000 iterations.

## A sample text generation:


      I remained that we were “New York Sirious.” I replied, “told out nearly
      steady—a pet within out of the malife.” My annual regions even such
      affects in successible incubation of any islands as remained in search, and
      with every topic of fiction by that icy conditions which was
      my will find among the ceptain villain from our group, and very
      unexpected by the subject proceeding in so very little company. When
      she determined its bendering upon the curiosity of the crooket.”

      “The details” have been conducted that the spectators we had designed
      to call its miscular expansion to be mercified that many thousand
      weeks, received a passage before me, must susceptibly to desire
      the features of the waistcoat was perfectly found in the existence
      of the first box of the hill, headlong in the sound of the muring
      interspersest time as we found it as well as they were, in a character,
      as Mpey’s house through the big scouthed into it. It is clear to
      leave it? My project then are as nearly containing, I could not have
      help to do. I should not spoke of rich.

      It is not impossible that, that they succeeded that they came no
      tone, like what with rumor, bodily-looked and mazellow in the plume
      which then all singularly sick and several more harassing than passed
      by three hours which we have attacquinted to call and usual, it
      seemed to attempt in the tapestry. Thus comparatively we arranged
      my own rummor, he could to attempt the expedient of his coast, to fall
      with a brutal child, without his seats at his narrative, an intellect
      shadowing, Peters several truth, about the body of the browst; and the
      end by the old muskets of the storeroom struck immediately in a table. I
      know nothing that is otherwise and earth in that bitter elbow solemnity
      of death, in no body, attempt, that some pervidom, that point of
      secret’s increases it desired to allow it, proceeded for a thing which
      came into cuttle and cupboard. Several persons afrostected in my
      case,, and is the most discovery and applied of inculstances; but, as
      well as I have before could imagine that have always called the
      most absurd scene with precessbure of the censer of his hands. Had
      the brow his part before the reason forth received chiming about
      his entire operations, and for the unjuel his eye, I’ll be without
      what person, the limit precipice of the investigation we have flown
      for a primitive rapidity, never fairly imagined to suspicious in
      search that one of the hulk thoroughly over our barrels he said; that the
      consideration of susceptibility was remarked upon the hatchway
      of the patronades behaved by such—so being unaudapted to allow the
      little bears of the alternatoge the stables which had been palpably
      beyond the friend of the progress, with the table, by a natural
      more influent bending to the bow-handkerchief is full expression. The Present
      was no almost what had been my father felled up in all in degrees
      to laughing whatever.

      “Had the former occasioned purely,” excepting the species of the
      earth.

      “Legrand,” said I, “and that it was not,” attained, “yet the jug
      of excitement unable that my public sense were so just at once. That
      is no establishment. Examined by way of —, I occupied, in its
      unnatural neater which it is still attributed a distance directly
      degree. The spring was altered, we found that keeping a beauty of
      course. My master could I believe that I could not yet take my
      visit. But in all is packings out before me I could see, in the
      very extreme extent, of the third extraordinary manner now somewhom
      belows.



## Requirements
* PyTorch 2.8.0

## Acknowledgments
Data source: [Edgar Allan Poe's corpus](http://www.gutenberg.org/files/25525/25525-h/25525-h.htm#2150link2H_4_0003)

This implementation of NanoGPT is based on [Karpathy's GPT class](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [an improved version of NanoGPT](https://github.com/karpathy/nanoGPT/tree/master)
