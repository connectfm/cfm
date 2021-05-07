package com.amplifyframework.datastore.generated.model;

import com.amplifyframework.core.model.annotations.HasMany;

import java.util.List;
import java.util.UUID;
import java.util.Objects;

import androidx.core.util.ObjectsCompat;

import com.amplifyframework.core.model.AuthStrategy;
import com.amplifyframework.core.model.Model;
import com.amplifyframework.core.model.ModelOperation;
import com.amplifyframework.core.model.annotations.AuthRule;
import com.amplifyframework.core.model.annotations.Index;
import com.amplifyframework.core.model.annotations.ModelConfig;
import com.amplifyframework.core.model.annotations.ModelField;
import com.amplifyframework.core.model.query.predicate.QueryField;

import static com.amplifyframework.core.model.query.predicate.QueryField.field;

/** This is an auto generated class representing the Song type in your schema. */
@SuppressWarnings("all")
@ModelConfig(pluralName = "Songs", authRules = {
  @AuthRule(allow = AuthStrategy.PUBLIC, operations = { ModelOperation.CREATE, ModelOperation.UPDATE, ModelOperation.DELETE, ModelOperation.READ })
})
public final class Song implements Model {
  public static final QueryField ID = field("id");
  public static final QueryField URI = field("uri");
  public static final QueryField NAME = field("name");
  public static final QueryField DURATION = field("duration");
  public static final QueryField PLAYLIST_SONGS_ID = field("playlistSongsId");
  private final @ModelField(targetType="ID", isRequired = true) String id;
  private final @ModelField(targetType="String", isRequired = true) String uri;
  private final @ModelField(targetType="String") String name;
  private final @ModelField(targetType="Artist") @HasMany(associatedWith = "songArtistsId", type = Artist.class) List<Artist> artists = null;
  private final @ModelField(targetType="Int") Integer duration;
  private final @ModelField(targetType="ID") String playlistSongsId;
  public String getId() {
      return id;
  }
  
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
  
  public String getPlaylistSongsId() {
      return playlistSongsId;
  }
  
  private Song(String id, String uri, String name, Integer duration, String playlistSongsId) {
    this.id = id;
    this.uri = uri;
    this.name = name;
    this.duration = duration;
    this.playlistSongsId = playlistSongsId;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Song song = (Song) obj;
      return ObjectsCompat.equals(getId(), song.getId()) &&
              ObjectsCompat.equals(getUri(), song.getUri()) &&
              ObjectsCompat.equals(getName(), song.getName()) &&
              ObjectsCompat.equals(getDuration(), song.getDuration()) &&
              ObjectsCompat.equals(getPlaylistSongsId(), song.getPlaylistSongsId());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getId())
      .append(getUri())
      .append(getName())
      .append(getDuration())
      .append(getPlaylistSongsId())
      .toString()
      .hashCode();
  }
  
  @Override
   public String toString() {
    return new StringBuilder()
      .append("Song {")
      .append("id=" + String.valueOf(getId()) + ", ")
      .append("uri=" + String.valueOf(getUri()) + ", ")
      .append("name=" + String.valueOf(getName()) + ", ")
      .append("duration=" + String.valueOf(getDuration()) + ", ")
      .append("playlistSongsId=" + String.valueOf(getPlaylistSongsId()))
      .append("}")
      .toString();
  }
  
  public static UriStep builder() {
      return new Builder();
  }
  
  /** 
   * WARNING: This method should not be used to build an instance of this object for a CREATE mutation.
   * This is a convenience method to return an instance of the object with only its ID populated
   * to be used in the context of a parameter in a delete mutation or referencing a foreign key
   * in a relationship.
   * @param id the id of the existing item this instance will represent
   * @return an instance of this model with only ID populated
   * @throws IllegalArgumentException Checks that ID is in the proper format
   */
  public static Song justId(String id) {
    try {
      UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
    } catch (Exception exception) {
      throw new IllegalArgumentException(
              "Model IDs must be unique in the format of UUID. This method is for creating instances " +
              "of an existing object with only its ID field for sending as a mutation parameter. When " +
              "creating a new object, use the standard builder method and leave the ID field blank."
      );
    }
    return new Song(
      id,
      null,
      null,
      null,
      null
    );
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(id,
      uri,
      name,
      duration,
      playlistSongsId);
  }
  public interface UriStep {
    BuildStep uri(String uri);
  }
  

  public interface BuildStep {
    Song build();
    BuildStep id(String id) throws IllegalArgumentException;
    BuildStep name(String name);
    BuildStep duration(Integer duration);
    BuildStep playlistSongsId(String playlistSongsId);
  }
  

  public static class Builder implements UriStep, BuildStep {
    private String id;
    private String uri;
    private String name;
    private Integer duration;
    private String playlistSongsId;
    @Override
     public Song build() {
        String id = this.id != null ? this.id : UUID.randomUUID().toString();
        
        return new Song(
          id,
          uri,
          name,
          duration,
          playlistSongsId);
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
     public BuildStep duration(Integer duration) {
        this.duration = duration;
        return this;
    }
    
    @Override
     public BuildStep playlistSongsId(String playlistSongsId) {
        this.playlistSongsId = playlistSongsId;
        return this;
    }
    
    /** 
     * WARNING: Do not set ID when creating a new object. Leave this blank and one will be auto generated for you.
     * This should only be set when referring to an already existing object.
     * @param id id
     * @return Current Builder instance, for fluent method chaining
     * @throws IllegalArgumentException Checks that ID is in the proper format
     */
    public BuildStep id(String id) throws IllegalArgumentException {
        this.id = id;
        
        try {
            UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
        } catch (Exception exception) {
          throw new IllegalArgumentException("Model IDs must be unique in the format of UUID.",
                    exception);
        }
        
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String id, String uri, String name, Integer duration, String playlistSongsId) {
      super.id(id);
      super.uri(uri)
        .name(name)
        .duration(duration)
        .playlistSongsId(playlistSongsId);
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
     public CopyOfBuilder duration(Integer duration) {
      return (CopyOfBuilder) super.duration(duration);
    }
    
    @Override
     public CopyOfBuilder playlistSongsId(String playlistSongsId) {
      return (CopyOfBuilder) super.playlistSongsId(playlistSongsId);
    }
  }
  
}
