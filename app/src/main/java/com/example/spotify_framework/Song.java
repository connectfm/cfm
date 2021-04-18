package com.example.spotify_framework;

import android.os.Parcel;
import android.os.Parcelable;

import java.util.ArrayList;
import java.util.List;

public class Song {
    private String id;
    private String uri;
    private String name;
    private int duration;
    private List<String> artistsList = new ArrayList<String>();
    private List<String> albumImages = new ArrayList<String>();
    private String artist_id;

    public Song(String id, String name) {
        this.name = name;
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public List<String> getArtists() {return artistsList;}


    public void setTitle() {
        this.name = name;
    }

    public List<String> getImages() {return albumImages;}

    public void setImage(String image) {
        if(albumImages == null)
            albumImages = new ArrayList<String>();
        albumImages.add(image);
    }

    public void setArtist(String artist) {
        if(artistsList == null)
            artistsList = new ArrayList<String>();
        albumImages.add(artist);
    }

}
