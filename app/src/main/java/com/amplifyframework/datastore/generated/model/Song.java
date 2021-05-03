package com.amplifyframework.datastore.generated.model;


import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the Song type in your schema. */
public final class Song {
  private final String uri;
  private final String name;
  private final List<Artist> artists;
  private final Integer duration;
  public String getUri() {
      return uri;
  }
  
  public String getName() {
      return name;
  }
  
  public List<Artist> getArtists() {
      return artists;
  }
  
  public Integer getDuration() {
      return duration;
  }
  
  private Song(String uri, String name, List<Artist> artists, Integer duration) {
    this.uri = uri;
    this.name = name;
    this.artists = artists;
    this.duration = duration;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Song song = (Song) obj;
      return ObjectsCompat.equals(getUri(), song.getUri()) &&
              ObjectsCompat.equals(getName(), song.getName()) &&
              ObjectsCompat.equals(getArtists(), song.getArtists()) &&
              ObjectsCompat.equals(getDuration(), song.getDuration());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getUri())
      .append(getName())
      .append(getArtists())
      .append(getDuration())
      .toString()
      .hashCode();
  }
  
  public static UriStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(uri,
      name,
      artists,
      duration);
  }
  public interface UriStep {
    BuildStep uri(String uri);
  }
  

  public interface BuildStep {
    Song build();
    BuildStep name(String name);
    BuildStep artists(List<Artist> artists);
    BuildStep duration(Integer duration);
  }
  

  public static class Builder implements UriStep, BuildStep {
    private String uri;
    private String name;
    private List<Artist> artists;
    private Integer duration;
    @Override
     public Song build() {
        
        return new Song(
          uri,
          name,
          artists,
          duration);
    }
    
    @Override
     public BuildStep uri(String uri) {
        Objects.requireNonNull(uri);
        this.uri = uri;
        return this;
    }
    
    @Override
     public BuildStep name(String name) {
        this.name = name;
        return this;
    }
    
    @Override
     public BuildStep artists(List<Artist> artists) {
        this.artists = artists;
        return this;
    }
    
    @Override
     public BuildStep duration(Integer duration) {
        this.duration = duration;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String uri, String name, List<Artist> artists, Integer duration) {
      super.uri(uri)
        .name(name)
        .artists(artists)
        .duration(duration);
    }
    
    @Override
     public CopyOfBuilder uri(String uri) {
      return (CopyOfBuilder) super.uri(uri);
    }
    
    @Override
     public CopyOfBuilder name(String name) {
      return (CopyOfBuilder) super.name(name);
    }
    
    @Override
     public CopyOfBuilder artists(List<Artist> artists) {
      return (CopyOfBuilder) super.artists(artists);
    }
    
    @Override
     public CopyOfBuilder duration(Integer duration) {
      return (CopyOfBuilder) super.duration(duration);
    }
  }
  
}
