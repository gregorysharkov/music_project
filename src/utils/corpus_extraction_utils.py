from typing import Dict, Iterable

import music21 as mu


def extract_notes(corpus: mu.stream.Score) -> Dict:
    """
    function extracts notes from a corpus
    Each part is extracted as a single dictionary element where name is the key, and the
    value is a list of notes, pauses and chords

    Args:
        corpus: a corpus to be parsed

    Returns:
        Dict
    """
    return_dict = {}
    for part in corpus.getElementsByClass(mu.stream.Part):
        part_name = part.partName
        if part_name in return_dict.keys():
            part_name = part_name + "_"
        part_notes = process_part(part)
        return_dict[part_name] = part_notes

    return return_dict


def process_part(part: mu.stream.Part) -> Iterable:
    """
    function preprocesses a given part

    Args:
        part: music 21 part to be processed

    Returns:
        a list of list. Each measure is isolated in a separate tuple
    """

    return [
        process_measure(measure) for measure in part.getElementsByClass(mu.stream.Measure)
    ]


def process_measure(measure: mu.stream.Measure) -> Iterable:
    """
    function preprocesses a measure depending on the type of the element

    Args:
        measure: a music 21 measure to be preprocessed

    Returns:
        an iterable containing a separate list for every note or chord or a pause
    """

    processors = {
        mu.note.Note: process_note,
        mu.note.Rest: process_pause,
        mu.chord.Chord: process_chord,
    }

    measures = []
    elements = measure.getElementsByClass(
        [mu.note.Note, mu.note.Rest, mu.chord.Chord])
    for element in elements:
        processor = processors.get(type(element), None)
        if processor:
            measures.append(processor(element=element, with_lengths=True))
        # else:
        #     print(type(element))
    return measures


def process_note(element: mu.note.Note, with_lengths: bool = True) -> Iterable:
    """
    function processes a note, it returns a tuple containing the pith and the duration

    Args:
        element: a note to be preprocessed. We will take the name with octave
        with_lenghts: a boolean flag indigating whether we want to extract lengths of notes

    Returns:
        a tuple with result of application of a proper processor (different for notes, pauses and chords)
    """

    return_list = [element.pitch.nameWithOctave]
    if with_lengths:
        return_list.append(element.duration.quarterLength)
    return return_list


def process_pause(element: mu.note.Rest, with_lengths: bool = True) -> Iterable:
    """function preprocesses a pause, it returns a tuple containing 'P' for pause and its duration"""

    return_list = ["P"]
    if with_lengths:
        return_list.append(element.duration.quarterLength)

    return tuple(return_list)


def process_chord(element: mu.chord.Chord, with_lengths: bool = True) -> Iterable:
    """
    function preprocesses a chord. it returns an iterable containing result of preprocessing
    of each note in the chord
    """

    return_list = []
    for note in element.notes:
        return_list.append(process_note(note, with_lengths=False)[0])

    if with_lengths:
        return_list = [return_list, element.duration.quarterLength]

    return tuple(return_list)
