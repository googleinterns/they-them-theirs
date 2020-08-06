function highlightDifference(oldElem, newElem) {
    // highlight the difference between the original sentence (oldElem) and the new sentence (newElem)

    // list of words in the original and new sentences
    let oldWords = get_words(oldElem),
        newWords = get_words(newElem);

    let removedWords = oldWords.filter(word => !newWords.includes(word)),  // list of removed words in original sentence
        addedWords = newWords.filter(word => !oldWords.includes(word));  // list of added words in new sentence

    var oldText = "",
        newText = "";

    // add the "removed" span tag to removed words in the original sentence
    for (var i=0; i < oldWords.length; i++) {
        if (removedWords.includes(oldWords[i])) {
            oldText += "<span class='removed'>" + oldWords[i] + "</span>";
        } else {
            oldText += oldWords[i];
        }
        if (i < oldWords.length - 1) {
            oldText += " "
        }
    }

    // add the "added" span tag to added words in the new sentence
    for (var i=0; i < newWords.length; i++) {
        if (addedWords.includes(newWords[i])) {
            newText += "<span class='added'>" + newWords[i] + "</span>";
        } else {
            newText += newWords[i];
        }
        if (i < newWords.length - 1) {
            newText += " "
        }
    }

    oldElem.html(oldText);
    newElem.html(newText);
}

function get_words(element) {
    // get the words of a sentence
    return element.text().split(' ');
}
