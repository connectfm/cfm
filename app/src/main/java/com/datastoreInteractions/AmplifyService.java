package com.datastoreInteractions;

import android.content.Context;
import android.util.Log;

import com.amplifyframework.AmplifyException;
import com.amplifyframework.core.Amplify;
import com.amplifyframework.core.model.query.Where;
import com.amplifyframework.core.model.temporal.Temporal;
import com.amplifyframework.datastore.AWSDataStorePlugin;
import com.amplifyframework.datastore.generated.model.Artist;
import com.amplifyframework.datastore.generated.model.Event;
import com.amplifyframework.datastore.generated.model.History;
import com.amplifyframework.datastore.generated.model.Location;
import com.amplifyframework.datastore.generated.model.Song;
import com.amplifyframework.datastore.generated.model.User;
import com.amplifyframework.datastore.generated.model.SongFeatures;
import com.spotifyFramework.VolleyCallBack;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class AmplifyService {

    private User queriedUser;

    public AmplifyService(Context context) {
        try {
            Amplify.addPlugin(new AWSDataStorePlugin());
            Amplify.configure(context);

            Log.d("CFM Amplify Interaction", "Amplify Initialized");
        } catch (AmplifyException e) {
            Log.e("CFM Amplify Interaction", "Could not initialize Amplify", e);
        }
    }

    public User getUser() {
        return queriedUser;
    }

    public void createUser(String id, String email) {
        User user = User.builder()
                .email(email)
                .build();

        Amplify.DataStore.save(user,
                res -> Log.d("CFM Amplify Interaction", "Successfully created new user"),
                error -> Log.e("CFM Amplify Interaction", "Error creating user", error));

    }

    public Location createLocation(Double lat, Double longi) {
        Location location = Location.builder()
                .lat(lat)
                .longi(longi)
                .build();
        return location;
    }

    public SongFeatures createSongFeatures(com.spotifyFramework.Song song) {
        Song s = createSong(song);
        SongFeatures features = SongFeatures.builder()
                .songId(song.getId())
                .danceability((double) song.getDanceability())
                .energy((double) song.getEnergy())
                .loudness((double) song.getLoudness())
                .speechiness((double) song.getSpeechiness())
                .acousticness((double) song.getAcousticness())
                .instrumentalness((double) song.getInstrumentalness())
                .liveness((double) song.getLiveness())
                .valence((double) song.getValence())
                .tempo((double) song.getTempo())
                .song(s)
                .build();
        return features;
    }

    public History createHistory(List<Event> events) {
        History history = History.builder()
                .lastUpdated(LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME))
                .events(events)
                .build();
        return history;
    }

    public Event createEvent(com.spotifyFramework.Song song, int rating) {
        Event event = Event.builder()
                .song(createSong(song))
                .rating(rating)
                .build();
        return event;
    }

    public Song createSong(com.spotifyFramework.Song song) {
        List<Artist> artists = new ArrayList<>();
        Map<String, String> artistList = song.getArtists();

        artistList.forEach((id,name) -> {
           artists.add(createArtist(id,name));
        });

        Song s = Song.builder()
                .uri(song.getUri())
                .name(song.getName())
                .artists(artists)
                .duration(song.getDuration().intValue())
                .build();
        return s;
    }

    public Artist createArtist(String id, String name) {
        Artist artist = Artist.builder()
                .artId(id)
                .name(name)
                .build();

        return artist;
    }

    public void queryUser(String email, VolleyCallBack callBack) {
        Amplify.DataStore.query(User.class, Where.matches(User.EMAIL.eq(email)), users -> {
            while(users.hasNext()) {
                queriedUser = users.next();
            }
        }, failure -> {
            Log.e("CFM Amplify Interaction", "Query Failed", failure);
        });
    }

}
