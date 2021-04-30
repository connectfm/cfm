package spotify_framework;

import android.os.Parcel;
import android.os.Parcelable;

import com.google.gson.Gson;

import java.util.ArrayList;
import java.util.List;

public class Song {
    private String id;
    private String uri;
    private String name;
    private int duration_ms;
    private List<String> artistsList = new ArrayList<String>();
    private List<String> albumImages = new ArrayList<String>();
    private String artist_id;
    private String albumName;

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

    public String getAlbum() {
        return albumName;
    }

    public List<String> getArtists() {return artistsList;}

    public void setAlbumName(String name) {
        this.albumName = name;
    }


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
        artistsList.add(artist);
    }

    public String toString() {
        Gson gson = new Gson();
        String res = gson.toJson(this);
        return res;
    }

    public String artistsToString(int limit) {
        StringBuilder sb = new StringBuilder();
        sb.append(getArtists().get(0));
        for(int i = 1; i < getArtists().size(); i++) {
            sb.append(", " + getArtists().get(i));
        }

        String res = sb.toString();
        if(res.length() > limit) {
            res = res.substring(0,limit) + "...";
        }
       return res;
    }

    public String getUri() {
        return uri;
    }
}
