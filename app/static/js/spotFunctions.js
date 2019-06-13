/*Universal Variables*/
var mood = "happy";
var genre = "edm";


function getGenre(form) {
    mood = document.getElementById("pred").innerHTML.toLowerCase();

    if (document.getElementById("genre1").checked == true) {
         genre = "edm";
    } else if (document.getElementById('genre2').checked == true) {
        genre = "rap";
    } else if (document.getElementById('genre3').checked == true) {
        genre = "pop";
    } else if (document.getElementById('genre4').checked == true) {
        genre = "country";
    } else if (document.getElementById('genre5').checked == true) {
        genre = "alternative";
    }

    alterMoodSetQuestion();
}


function alterMoodSetQuestion() {
    document.getElementById("p3Question").innerHTML = "Would you like to listen to "
        + mood + " " + genre + " music?";

    if (mood == "happy") {
        document.getElementById('alter').getElementsByTagName('label')[0].innerHTML = "Yes, remain happy.";
        document.getElementById('mood1').checked = true;
    } else if (mood == "sad") {
        document.getElementById('alter').getElementsByTagName('label')[1].innerHTML = "Yes, remain sad.";
        document.getElementById('mood1').checked = false;
        document.getElementById('mood2').checked = true;
    } else if (mood == "angry") {
        document.getElementById('alter').getElementsByTagName('label')[2].innerHTML = "Yes, remain angry.";
        document.getElementById('mood1').checked = false;
        document.getElementById('mood3').checked = true;
    } else if (mood == "surprised") {
        document.getElementById('alter').getElementsByTagName('label')[3].innerHTML = "Yes, I'd like to be surprised.";
        document.getElementById('mood1').checked = false;
        document.getElementById('mood4').checked = true;
    }

    document.getElementById('PageThree').style.visibility = 'visible';
}

function alterMood() {
    if (document.getElementById('mood1').checked == true) {
        mood = "happy";
    } else if (document.getElementById('mood2').checked == true) {
        mood = "sad";
    } else if (document.getElementById('mood3').checked == true) {
        mood = "angry";
    } else if (document.getElementById('mood4').checked == true) {
        mood = "surprised";
    }

    getPlaylist();
    document.getElementById('PageFour').style.visibility = 'visible';
}


function getPlaylist() {
    if (mood == "happy") {
        if (genre == "edm") {
		document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DX6VdMW310YC7?si=if2FHhFnRWO0tFcPwVAV8Q"
	    } else if (genre == "rap") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/2BnJlF2E9UgIyuL1qlNjjG?si=GTp552wKTfedNhg22KHqMQ"
            } else if (genre == "country") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DWTkxQvqMy4WW?si=RKFm7gREQqSxDZDRd3KpqQ"
            } else if (genre == "pop") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC?si=QKkuTurnRS2nZXkl75utIg"
            } else if (genre == "alternative") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/5CDM8rVt27yt09BO5wA5oP?si=dLVDN59NSTanreABrVGjzw"
            }
        }
        else if (mood == "sad") {
            if (genre == "edm") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DWVrtsSlLKzro?si=oBJUqxjSQJiPIuiVGrIXuw"
            } else if (genre == "rap") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/4XpY2PtCUqzhddUnFEbckt?si=SST3A5IRRvSu7xqhffLBoQ"
            } else if (genre == "country") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DXdmMcgFhLQ8u?si=a2XS0BFGTO-tqDmDq3LOWQ"
            } else if (genre == "pop") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DX3YSRoSdA634?si=KIcRrJ_0SA-TIbLvCJ34pg"
            } else if (genre == "alternative") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/5tPAmcKqoBrGChGO6lMuHu?si=tunyjGu8Q92WOoeNgYluMA"
            }
        }
        else if (mood == "angry") {
            if (genre == "edm") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DX3ND264N08pv?si=lOVuq3rQSnmxft5zRed_Vg"
            } else if (genre == "rap") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/62cdYStZNC1u3l6Cu9phP9?si=Q7m2SsEwQ-2UQzigsrN1lA"
            } else if (genre == "country") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/37i9dQZF1DXc0fQ4sGOnVJ?si=v1KpAR61R9OuTCegBzltZw"
            } else if (genre == "pop") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/4ca6MwaTAiz852ZffCSWRS?si=DCGL86GsQ0ah12PrhsJG9w"
            } else if (genre == "alternative") {
                document.getElementById('spotIframe').src = "https://open.spotify.com/embed/playlist/6UkcmQf7tYlGKX18IL8cyd?si=5oZ4NmJfQhCZIV_uMCw6AQ"
            }
        }
        else {
            document.getElementById('spotIframe').src = "https://open.spotify.com/embed/user/1240281655/playlist/0GrlAcPQBhh0HDXLODAwuv"
        }
}