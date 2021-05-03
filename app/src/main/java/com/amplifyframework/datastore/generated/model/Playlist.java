package com.amplifyframework.datastore.generated.model;


import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the Playlist type in your schema. */
public final class Playlist {
  private final String playlist_id;
  private final String name;
  private final List<Song> songs;
  public String getPlaylistId() {
      return playlist_id;
  }
  
  public String getName() {
      return name;
  }
  
  public List<Song> getSongs() {
      return songs;
  }
  
  private Playlist(String playlist_id, String name, List<Song> songs) {
    this.playlist_id = playlist_id;
    this.name = name;
    this.songs = songs;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Playlist playlist = (Playlist) obj;
      return ObjectsCompat.equals(getPlaylistId(), playlist.getPlaylistId()) &&
              ObjectsCompat.equals(getName(), playlist.getName()) &&
              ObjectsCompat.equals(getSongs(), playlist.getSongs());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getPlaylistId())
      .append(getName())
      .append(getSongs())
      .toString()
      .hashCode();
  }
  
  public static PlaylistIdStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(playlist_id,
      name,
      songs);
  }
  public interface PlaylistIdStep {
    NameStep playlistId(String playlistId);
  }
  

  public interface NameStep {
    BuildStep name(String name);
  }
  

  public interface BuildStep {
    Playlist build();
    BuildStep songs(List<Song> songs);
  }
  

  public static class Builder implements PlaylistIdStep, NameStep, BuildStep {
    private String playlist_id;
    private String name;
    private List<Song> songs;
    @Override
     public Playlist build() {
        
        return new Playlist(
          playlist_id,
          name,
          songs);
    }
    
    @Override
     public NameStep playlistId(String playlistId) {
        Objects.requireNonNull(playlistId);
        this.playlist_id = playlistId;
        return this;
    }
    
    @Override
     public BuildStep name(String name) {
        Objects.requireNonNull(name);
        this.name = name;
        return this;
    }
    
    @Override
     public BuildStep songs(List<Song> songs) {
        this.songs = songs;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String playlistId, String name, List<Song> songs) {
      super.playlistId(playlistId)
        .name(name)
        .songs(songs);
    }
    
    @Override
     public CopyOfBuilder playlistId(String playlistId) {
      return (CopyOfBuilder) super.playlistId(playlistId);
    }
    
    @Override
     public CopyOfBuilder name(String name) {
      return (CopyOfBuilder) super.name(name);
    }
    
    @Override
     public CopyOfBuilder songs(List<Song> songs) {
      return (CopyOfBuilder) super.songs(songs);
    }
  }
  
}
