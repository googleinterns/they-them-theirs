function highlight(oldElem, newElem) {
    var oldText = oldElem.text().split(' '),
        newText = newElem.text().split(' '),
        added = '',
        removed = '';

    newText.forEach(function(val, i){
        if (val != oldText[i]) {
            added += "<span class='added'>" + val + "</span>";
            removed += "<span class='removed'>" + oldText[i] + "</span>";
        } else {
            added += val;
            removed += oldText[i];
        }
        added += " ";
        removed += " ";
    });

    oldElem.html(removed);
    newElem.html(added);
}