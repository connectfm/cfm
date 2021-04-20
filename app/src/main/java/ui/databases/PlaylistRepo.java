package ui.databases;

import android.app.Application;

import androidx.lifecycle.LiveData;

import spotify_framework.Playlist;

import java.util.List;

public class PlaylistRepo {

    private LiveData<List<Playlist>> playlists;
    public void insert(Playlist p){}
    public void delete(Playlist p){}
    public void update(Playlist p){}

    public LiveData<List<Playlist>> getAllPlaylists(){
        return playlists;
         }
    public void setAllPlaylists(List<Playlist> plists){}
    public PlaylistRepo(Application app) {}
    public PlaylistRepo(){}

}
