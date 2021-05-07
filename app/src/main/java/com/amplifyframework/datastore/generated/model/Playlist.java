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

/** This is an auto generated class representing the Playlist type in your schema. */
@SuppressWarnings("all")
@ModelConfig(pluralName = "Playlists", authRules = {
  @AuthRule(allow = AuthStrategy.PUBLIC, operations = { ModelOperation.CREATE, ModelOperation.UPDATE, ModelOperation.DELETE, ModelOperation.READ })
})
public final class Playlist implements Model {
  public static final QueryField ID = field("id");
  public static final QueryField PLAYLIST_ID = field("playlist_id");
  public static final QueryField NAME = field("name");
  private final @ModelField(targetType="ID", isRequired = true) String id;
  private final @ModelField(targetType="String", isRequired = true) String playlist_id;
  private final @ModelField(targetType="String", isRequired = true) String name;
  private final @ModelField(targetType="Song") @HasMany(associatedWith = "playlistSongsId", type = Song.class) List<Song> songs = null;
  public String getId() {
      return id;
  }
  
  public String getPlaylistId() {
      return playlist_id;
  }
  
  public String getName() {
      return name;
  }
  
  public List<Song> getSongs() {
      return songs;
  }
  
  private Playlist(String id, String playlist_id, String name) {
    this.id = id;
    this.playlist_id = playlist_id;
    this.name = name;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Playlist playlist = (Playlist) obj;
      return ObjectsCompat.equals(getId(), playlist.getId()) &&
              ObjectsCompat.equals(getPlaylistId(), playlist.getPlaylistId()) &&
              ObjectsCompat.equals(getName(), playlist.getName());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getId())
      .append(getPlaylistId())
      .append(getName())
      .toString()
      .hashCode();
  }
  
  @Override
   public String toString() {
    return new StringBuilder()
      .append("Playlist {")
      .append("id=" + String.valueOf(getId()) + ", ")
      .append("playlist_id=" + String.valueOf(getPlaylistId()) + ", ")
      .append("name=" + String.valueOf(getName()))
      .append("}")
      .toString();
  }
  
  public static PlaylistIdStep builder() {
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
  public static Playlist justId(String id) {
    try {
      UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
    } catch (Exception exception) {
      throw new IllegalArgumentException(
              "Model IDs must be unique in the format of UUID. This method is for creating instances " +
              "of an existing object with only its ID field for sending as a mutation parameter. When " +
              "creating a new object, use the standard builder method and leave the ID field blank."
      );
    }
    return new Playlist(
      id,
      null,
      null
    );
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(id,
      playlist_id,
      name);
  }
  public interface PlaylistIdStep {
    NameStep playlistId(String playlistId);
  }
  

  public interface NameStep {
    BuildStep name(String name);
  }
  

  public interface BuildStep {
    Playlist build();
    BuildStep id(String id) throws IllegalArgumentException;
  }
  

  public static class Builder implements PlaylistIdStep, NameStep, BuildStep {
    private String id;
    private String playlist_id;
    private String name;
    @Override
     public Playlist build() {
        String id = this.id != null ? this.id : UUID.randomUUID().toString();
        
        return new Playlist(
          id,
          playlist_id,
          name);
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
    private CopyOfBuilder(String id, String playlistId, String name) {
      super.id(id);
      super.playlistId(playlistId)
        .name(name);
    }
    
    @Override
     public CopyOfBuilder playlistId(String playlistId) {
      return (CopyOfBuilder) super.playlistId(playlistId);
    }
    
    @Override
     public CopyOfBuilder name(String name) {
      return (CopyOfBuilder) super.name(name);
    }
  }
  
}
