package com.amplifyframework.datastore.generated.model;


import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the Artist type in your schema. */
public final class Artist {
  private final String art_id;
  private final String name;
  public String getArtId() {
      return art_id;
  }
  
  public String getName() {
      return name;
  }
  
  private Artist(String art_id, String name) {
    this.art_id = art_id;
    this.name = name;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Artist artist = (Artist) obj;
      return ObjectsCompat.equals(getArtId(), artist.getArtId()) &&
              ObjectsCompat.equals(getName(), artist.getName());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getArtId())
      .append(getName())
      .toString()
      .hashCode();
  }
  
  public static ArtIdStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(art_id,
      name);
  }
  public interface ArtIdStep {
    BuildStep artId(String artId);
  }
  

  public interface BuildStep {
    Artist build();
    BuildStep name(String name);
  }
  

  public static class Builder implements ArtIdStep, BuildStep {
    private String art_id;
    private String name;
    @Override
     public Artist build() {
        
        return new Artist(
          art_id,
          name);
    }
    
    @Override
     public BuildStep artId(String artId) {
        Objects.requireNonNull(artId);
        this.art_id = artId;
        return this;
    }
    
    @Override
     public BuildStep name(String name) {
        this.name = name;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String artId, String name) {
      super.artId(artId)
        .name(name);
    }
    
    @Override
     public CopyOfBuilder artId(String artId) {
      return (CopyOfBuilder) super.artId(artId);
    }
    
    @Override
     public CopyOfBuilder name(String name) {
      return (CopyOfBuilder) super.name(name);
    }
  }
  
}
